import os
import cv2
import numpy as np
import argparse
import open3d as o3d
import torch
from urllib.request import urlretrieve

from util.visualization import get_colored_point_cloud_feature, get_colored_point_cloud_feature_pc
from util.misc import extract_features, extract_features_mdfy
from tqdm import tqdm
from model.resunet import ResUNetBN2C
from tool.helpers import compute_pixel_mask


"""读取一帧RGB-D帧，转成点云并提取每个点的feature"""

seq_name = "scene0011_00"
frame_id = 0

seq_dir = "/media/javens/TYJ_lab1T/Dataset/ScanNet/all_sens/scans/%s" % seq_name
dst_h = 480
dst_w = 640
depth_scale = 1000.
color_image_path = os.path.join(seq_dir, "output/color", "%d.jpg" % frame_id)
depth_image_path = os.path.join(seq_dir, "output/depth", "%d.png" % frame_id)
intrinsic_path = os.path.join(seq_dir, "output/intrinsic/intrinsic_depth.txt")


# @brief: get Gaussian poincloud for a given RGB-D frame
# @param color: Tensor(h, w, 3);
# @param depth: Tensor(h, w);
# @param intrinsics: Tensor(3, 3);
# @param pose_c2w: Tensor(4, 4);
# @param transform_pts: whether transform point from Camera coordinates to World coordinates, bool;
# @param mask: Tensor(h * w), dtype=bool;
# @param compute_mean_sq_dist: bool;
# @param mean_sq_dist_method:
#-@return point_cld: [x, y, z, R, G, B], Tensor(valid_pts_num, 6).
def get_pointcloud(color, depth, intrinsics, pose_c2w, transform_pts=True, mask=None):
    width, height = color.shape[1], color.shape[0]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).float(), torch.arange(height).float(), indexing="xy")
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth.reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)  # Tensor(h * w, 3)
    if transform_pts:  # default
        pix_ones = torch.ones(height * width, 1).float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        pts = (pose_c2w @ pts4.T).T[:, :3]  # point world coordinates, Tensor(h * w, 3)
    else:
        pts = pts_cam
    cols = color.reshape(-1, 3)  # Tensor(h, w, 3) -> Tensor(h * w, 3)

    if mask is not None:
        depth_z = depth_z[mask]
        pts = pts[mask]
        cols = cols[mask]

    point_cld = torch.cat((pts, cols), -1)  # Tensor(h * w, 6), [x, y, z, R, G, B]
    return point_cld

def rgbd_to_pc(rgb_image, depth_image, K, pose_c2w, mask):
    depth_mask = (depth_img > 0).reshape(-1)
    mask = mask & depth_mask
    pc_frame = get_pointcloud(rgb_image, depth_image, K, pose_c2w, mask=mask)  # Tensor(valid_pts_num, 6)
    pts_xyz = pc_frame[:, :3].numpy().astype("float64")
    pts_color = pc_frame[:, 3:].numpy().astype("float64")

    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(pts_xyz)
    vis_pcd.colors = o3d.utility.Vector3dVector(pts_color)
    return vis_pcd


def demo(config, input_pcd, output_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(config.model)
    model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)  # feature_dim = 16
    # model = ResUNetBN2C(1, 32, normalize_feature=True, conv1_kernel_size=3, D=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)

    pcd = input_pcd
    pts_xyz = np.array(pcd.points)  # original 3D coordinates of each input point

    # *** firstly input points will be downsampled(voxelized), then voxelized points will be sent to extract per-point feature
    # @return xyz_down: voxelized point coordinates, ndarray(ds_pts_num, 3);5
    # @return feature: per-point feature of each voxelized point, Tensor(ds_pts_num, feature_dim);
    # @return indices_inverse: mapping table. For each original point, its corresponding voxelized point, Tensor(pts_num, );
    xyz_down, feature, indices_inverse = extract_features_mdfy(model, xyz=pts_xyz, voxel_size=config.voxel_size, device=device, skip_check=True)

    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(xyz_down)

    # vis_pcd = get_colored_point_cloud_feature(vis_pcd, feature.detach().cpu().numpy(), config.voxel_size)  # colored pc以o3d.geometry.TriangleMesh形式展示
    colorized_pcd = get_colored_point_cloud_feature_pc(vis_pcd, feature.detach().cpu().numpy())  # colored pc以o3d.geometry.PointCloud形式展示

    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        o3d.io.write_point_cloud(output_path, colorized_pcd)

    o3d.visualization.draw_geometries([colorized_pcd])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        default='redkitchen-20.ply',
        type=str,
        help='path to a pointcloud file')
    parser.add_argument(
        '-m',
        '--model',
        default='model/ResUNetBN2C-16feat-3conv.pth',
        type=str,
        help='path to latest checkpoint (default: None)')
    parser.add_argument(
        '--voxel_size',
        default=0.02,
        type=float,
        help='voxel size to preprocess point cloud')
    parser.add_argument(
        '--output_path',
        default="./test/output_pc.ply",
        type=str,
        help='output path of pc')

    # Step 1: read RGB-D frame and intrinsic
    color_img = cv2.imread(color_image_path)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # BGR --> RGB
    if color_img.shape[0] != dst_h or color_img.shape[1] != dst_w:
        color_img = cv2.resize(color_img, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    color_img = torch.from_numpy(color_img) / 255

    depth_img = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    depth_img = depth_img.astype("float32") / depth_scale
    depth_img = torch.from_numpy(depth_img)

    cam_intrinsic = np.loadtxt(intrinsic_path)[:3, :3].astype("float32")  # ndarray(3, 3), dtype=float32
    cam_intrinsic = torch.from_numpy(cam_intrinsic)
    pose_c2w = torch.eye(4).float()

    pixel_mask = compute_pixel_mask(dst_h, dst_w, 4, 4)
    pcd = rgbd_to_pc(color_img, depth_img, cam_intrinsic, pose_c2w, pixel_mask)


    config = parser.parse_args()
    demo(config, pcd, config.output_path)
