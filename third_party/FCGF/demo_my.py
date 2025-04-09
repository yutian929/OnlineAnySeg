import os
import numpy as np
import argparse
import open3d as o3d
from urllib.request import urlretrieve
from util.visualization import get_colored_point_cloud_feature, get_colored_point_cloud_feature_pc
from util.misc import extract_features, extract_features_mdfy
from tqdm import tqdm
from model.resunet import ResUNetBN2C

import torch

if not os.path.isfile('model/ResUNetBN2C-16feat-3conv.pth'):
    print('Downloading weights...')
    urlretrieve("https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth", 'ResUNetBN2C-16feat-3conv.pth')

if not os.path.isfile('test/redkitchen-20.ply'):
    print('Downloading a mesh...')
    urlretrieve("https://node1.chrischoy.org/data/publications/fcgf/redkitchen-20.ply", 'redkitchen-20.ply')


def demo(config, output_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(config.model)
    model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)  # feature_dim = 16
    # model = ResUNetBN2C(1, 32, normalize_feature=True, conv1_kernel_size=3, D=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)

    pcd = o3d.io.read_point_cloud(config.input)
    pts_xyz = np.array(pcd.points)  # original 3D coordinates of each input point

    # *** firstly input points will be downsampled(voxelized), then voxelized points will be sent to extract per-point feature
    # @return xyz_down: voxelized point coordinates, ndarray(ds_pts_num, 3);5
    # @return feature: per-point feature of each voxelized point, Tensor(ds_pts_num, feature_dim);
    # @return indices_inverse: mapping table. For each original point, its corresponding voxelized point, Tensor(pts_num, );
    xyz_down, feature, indices_inverse = extract_features_mdfy(model, xyz=pts_xyz, voxel_size=config.voxel_size, device=device, skip_check=True)

    raw_pts_feature = feature[indices_inverse]

    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(pts_xyz)

    # vis_pcd = get_colored_point_cloud_feature(vis_pcd, feature.detach().cpu().numpy(), config.voxel_size)  # colored pc以o3d.geometry.TriangleMesh形式展示
    colorized_pcd = get_colored_point_cloud_feature_pc(vis_pcd, raw_pts_feature.detach().cpu().numpy())  # colored pc以o3d.geometry.PointCloud形式展示

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
        default=0.03,
        type=float,
        help='voxel size to preprocess point cloud')
    parser.add_argument(
        '--output_path',
        default="./test/output_pc.ply",
        type=str,
        help='output path of pc')

    config = parser.parse_args()
    demo(config, config.output_path)
