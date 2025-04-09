import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.ops import ball_query
from collections import deque
import numpy as np
from scipy.ndimage import label
import open3d as o3d
import cv2
from torch.autograd import Variable
from math import exp

from tool.sampling_helper import pixel_rc_to_indices, sample_pixels_uniformly
from tool.visualization_helpers import get_new_pallete


def get_depth_mask(depth_tensor, dist_far=10.):
    # depth_tensor = torch.from_numpy(depth).cuda()
    depth_mask = torch.logical_and(depth_tensor > 0, depth_tensor < dist_far).reshape(-1)
    return depth_mask


def compose_transformations(trans_01, trans_12):
    # decompose to R, t
    rmat_01 = trans_01[..., :3, :3]  # Nx3x3
    rmat_12 = trans_12[..., :3, :3]  # Nx3x3
    tvec_01 = trans_01[..., :3, -1:]  # Nx3x1
    tvec_12 = trans_12[..., :3, -1:]  # Nx3x1

    # compute R', t'
    rmat_02 = torch.matmul(rmat_01, rmat_12)
    tvec_02 = torch.matmul(rmat_01, tvec_12) + tvec_01

    # pack output tensor
    trans_02 = torch.zeros_like(trans_01)
    trans_02[..., :3, 0:3] += rmat_02
    trans_02[..., :3, -1:] += tvec_02
    trans_02[..., -1, -1:] += 1.0
    return trans_02


# @param depth: ndarray(H, W)
# @param pinhole_cam_intrinsic: o3d.camera.PinholeCameraIntrinsic obj
# @param pose_c2w: ndarray(4, 4);
#-@return: pts with valid depth, ndarray(n, 4).
def backproject_pts(depth, pinhole_cam_intrinsic, pose_c2w, dist_far=10.):
    depth = o3d.geometry.Image(depth)
    pcld = o3d.geometry.PointCloud.create_from_depth_image(depth, pinhole_cam_intrinsic, depth_scale=1, depth_trunc=dist_far)
    pcld.transform(pose_c2w)
    trans_pts = np.asarray(pcld.points)
    return trans_pts


# @brief: for a given square matrix, set all its diagonal elements to given value;
# @param square_mat: Tensor(n, n);
# @param diag_value: float;
#-return: Tensor(n, n).
def set_mat_diag(square_mat, diag_value=0.):
    assert square_mat.shape[0] == square_mat.shape[1], "Input matrix is not square matrix!"

    I = torch.eye(square_mat.shape[0], dtype=torch.float32, device=square_mat.device)
    new_square_mat = square_mat.float() - square_mat * I
    new_square_mat += (diag_value * I)
    return new_square_mat.to(square_mat)


def connected_components_with_sizes(mask):
    """
    判断 mask 是否有多个连通分量，并返回每个连通分量的大小
    mask: 2D tensor (H, W)，单通道，值为 0 或 1
    return:
        - True/False 是否有多个连通分量
        - 连通分量大小列表（不包括背景）
    """
    # 确保 mask 是二值化的
    mask = (mask > 0).int()

    # 转换为 NumPy 数组
    mask_np = mask.cpu().numpy()

    # 使用 scipy 的 label 函数来标记连通分量
    labeled_array, num_features = label(mask_np)

    # 计算每个连通分量的大小
    component_sizes = []
    for component_id in range(1, num_features + 1):
        size = np.sum(labeled_array == component_id)
        component_sizes.append(size)

    return num_features, component_sizes


# @brief: given a pointcloud, remove noise points from it (only maximal connected component is kept)
# @param
def denoise(pcd, eps=0.04, remove_percent=0.2):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=4)) + 1  # -1 for noise
    mask = np.zeros(len(labels), dtype=bool)
    count = np.bincount(labels)  # 每个label_id在labels中的出现次数
    max_index = np.argmax(count)

    mask[labels == max_index] = True
    remain_index = np.where(mask)[0]
    pcd = pcd.select_by_index(remain_index)

    pcd, index = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    remain_index = remain_index[index]
    return pcd, remain_index


# @brief: given a pointcloud, remove noise points from it (remove small clusters)
def denoise2(pcd, eps=0.04, remove_percent=0.2, max_rm_pts_num=200):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=4)) + 1  # -1 for noise
    mask = np.ones(len(labels), dtype=bool)
    count = np.bincount(labels)  # 每个label_id在labels中的出现次数

    # remove component with less than threshold% points
    for i in range(len(count)):
        pts_threshold = min(max_rm_pts_num, remove_percent * len(labels))
        if count[i] < pts_threshold:
            mask[labels == i] = False

    remain_index = np.where(mask)[0]
    pcd = pcd.select_by_index(remain_index)

    pcd, index = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    remain_index = remain_index[index]
    return pcd, remain_index


def crop_scene_points(mask_points, scene_points, pad_dist=0.):
    x_min, x_max = torch.min(mask_points[:, 0]), torch.max(mask_points[:, 0])
    y_min, y_max = torch.min(mask_points[:, 1]), torch.max(mask_points[:, 1])
    z_min, z_max = torch.min(mask_points[:, 2]), torch.max(mask_points[:, 2])

    selected_point_mask = ( (scene_points[:, 0] > x_min - pad_dist) & (scene_points[:, 0] < x_max + pad_dist) &
                            (scene_points[:, 1] > y_min - pad_dist) & (scene_points[:, 1] < y_max + pad_dist) &
                            (scene_points[:, 2] > z_min - pad_dist) & (scene_points[:, 2] < z_max + pad_dist) )
    selected_point_ids = torch.where(selected_point_mask)[0]
    cropped_scene_points = scene_points[selected_point_ids]
    return cropped_scene_points, selected_point_ids


# @brief: compute intersection of 2 given 1-D tensor, i.e.: a ^ b;
# @param a: Tensor(m, );
# @param b: Tensor(n, );
#-@return: Tensor(intersect_num, ).
def compute_intersection(a, b):
    tensor_isin = torch.isin(a, b)  # whether each element of a is in b
    return a[tensor_isin]


# @brief: compute intersection of 2 given 1-D tensor, i.e.: a ^ b;
# @param a: Tensor(m, );
# @param b: Tensor(n, );
#-@return: int.
def compute_intersection_size(a, b):
    tensor_isin = torch.isin(a, b)  # whether each element of a is in b
    intersection_count = tensor_isin.sum().item()  # 交集的数量
    return intersection_count


# @brief: compute intersection of 2 given m-D tensor, i.e.: a ^ b;
# @param a: Tensor(m, d);
# @param b: Tensor(n, d);
#-@return intersect_elements: Tensor(intersect_element_num, d);
#-@return a_in_b: Tensor(m, ), dtype=bool;
def compute_intersection_md2(a, b):
    a_expanded = a.unsqueeze(1).expand(-1, b.size(0), -1)  # Tensor(m, n, d)
    b_expanded = b.unsqueeze(0).expand(a.size(0), -1, -1)  # Tensor(m, n, d)

    match_matrix = (a_expanded == b_expanded).all(dim=-1)  # Tensor(m, n)
    a_in_b = match_matrix.any(dim=1)   # 如果某行有任意元素为True，说明A中对应坐标在B中出现
    intersect_elements = a[a_in_b]
    return intersect_elements, a_in_b


# @brief: compute intersection of 2 given m-D tensor, i.e.: a ^ b;
# @param a: Tensor(m, d);
# @param b: Tensor(n, d);
#-@return intersect_elements: Tensor(intersect_element_num, d);
#-@return a_in_b: Tensor(m, ), dtype=bool;
def compute_intersection_md(a, b, batch_size=10000):
    m, _ = a.shape
    n, _ = b.shape
    a_in_b_mask = torch.zeros_like(a[:, 0]).bool()

    # 将A分批处理
    for i in range(0, m, batch_size):
        a_batch = a[i: i + batch_size]

        # 对每一批次的A，分别和B的所有分批比较
        for j in range(0, n, batch_size):
            b_batch = b[j: j + batch_size]

            a_expanded = a_batch.unsqueeze(1)  # 形状变为 (m_i, 1, 3)
            b_expanded = b_batch.unsqueeze(0)  # 形状变为 (1, n_i, 3)

            match_matrix = (a_expanded == b_expanded).all(dim=-1)  # 计算当前批次的相等性, 得到 (m_i, n_i) 的布尔矩阵
            a_in_b = match_matrix.any(dim=1)  # 如果某行有任意元素为True，说明A中对应坐标在B中出现
            current_indices = torch.where(a_in_b)[0]

            # 将当前批次的索引转换回全局索引
            if current_indices.numel() > 0:
                global_indices_a = current_indices + i
                a_in_b_mask[global_indices_a] = True

    intersect_num = torch.count_nonzero(a_in_b_mask).item()
    if intersect_num > 0:
        intersect_elements = a[a_in_b_mask]  # 提取A中的交集点
    else:
        intersect_elements = torch.empty((0, 3), dtype=a.dtype)  # 如果没有交集，返回空张量

    return intersect_elements, a_in_b_mask


# @brief: compute complementary set of 2 given 1-D tensor, i.e.: a - b;
# @param a: Tensor(m, );
# @param b: Tensor(n, );
def compute_complementary(a, b):
    tensor_not_in = torch.isin(a, b, invert=True)  # whether each element of a is NOT in b
    return a[tensor_not_in]

# @brief: compute complementary set of 2 given 1-D tensor, i.e.: a - b;
# @param a: Tensor(m, );
# @param b: Tensor(n, );
#-@return a_minus_b: Tensor(m', );
#-@return tensor_not_in: Tensor(m, ), dtype=bool.
def compute_complementary_w_mask(a, b):
    if b is None:
        tensor_not_in = torch.ones_like(a, dtype=torch.bool)
        a_minus_b = a
    else:
        tensor_not_in = torch.isin(a, b, invert=True)  # whether each element of a is NOT in b
        a_minus_b = a[tensor_not_in]
    return a_minus_b, tensor_not_in

# @brief: for each column in a 2D bool tensor, keep only 1 True elements(w minimal row_ID), and set other True to False;
# @param mask_tensor: bool tensor, Tensor(m, n), dtype=bool;
#-@return: Tensor(m, n), dtype=bool.
def keep_min_rows(mask_tensor):
    cumsum = mask_tensor.cumsum(dim=0)  # cumulative sum for each col, Tensor(m, n), dtype=int
    first_true_mask = (cumsum == 1)  # 仅保留每列第一个 True 的位置

    mask_tensor_new = mask_tensor & first_true_mask
    return mask_tensor_new

# @brief: given a batch of point coordinates and their per-point features, aggregate these features according to their distances to centroid;
# @param pts_xyz: Tensor(n, 3);
# @param pts_features: Tensor(b, feat_dim);
#-@return: Tensor(feat_dim, ).
def aggregate_pts_feature(pts_xyz, pts_features):
    # Step 1: compute per-point weight
    centroid_xyz = torch.mean(pts_xyz, dim=0)  # centroid coordinate
    distances = torch.norm(pts_xyz - centroid_xyz, dim=1)

    weights = 1.0 / (distances + 1e-8)
    normalized_weights = weights / torch.sum(weights)

    # Step 2: compute weighted sum of all input features
    weighted_features = pts_features * normalized_weights.unsqueeze(1)
    aggregated_feature = torch.sum(weighted_features, dim=0)
    return aggregated_feature

# @brief: given two 1-D tensor, add elements in A to B. If an element in A is already in B, skip it
# @param a: Tensor(m, );
# @param b: Tensor(n, );
def add_wo_redundant(a, b):
    if a is None or a.shape[0]==0:
        return b
    if b is None:
        return a

    tensor_not_in = torch.isin(a, b, invert=True)  # whether each element of a is NOT in b
    if torch.count_nonzero(tensor_not_in) > 0:
        new_tensor = torch.cat([b, a[tensor_not_in]], dim=0)
    else:
        new_tensor = b
    return new_tensor

# @brief: do non-linear mapping for an input tensor by exponential function;
# @param x: input tensor
# @param l: control parameter,
#-@return: mapped tensor.
def nonlinear_mapping(x, min_value=0.7, max_value=1.0, k=0.1):
    x = torch.clamp(x, min=min_value, max=max_value)

    # 使用指数函数进行非线性映射
    numerator = 1 - torch.exp(-k * (x - min_value))
    denominator = 1 - exp(-k * (max_value - min_value))

    mapped_x = numerator / denominator
    return mapped_x

# @brief: given a batch of query points, query their corresponding neighbors in scene poincloud;
# @param query_points: Tensor(n1, 3);
# @param scene_points: Tensor(n2, 3);
#-@return: each query point's neighbor in scene_points(point_id), Tensor(n1, k).
def query_neighbors(query_points, scene_points, k=10, radius=0.1, device="cuda:0"):
    length1 = torch.full((1, ), fill_value=query_points.shape[0], dtype=torch.int64, device=device)
    length2 = torch.full((1, ), fill_value=scene_points.shape[0], dtype=torch.int64, device=device)
    p1 = query_points.unsqueeze(0)
    p2 = scene_points.unsqueeze(0)
    _, neighbor_in_scene_pcd, _ = ball_query(p1, p2, length1, length2, K=k, radius=radius)
    return neighbor_in_scene_pcd[0]


# @brief: for each point in 'valid_points', query its top K nearest neighbors in 'scene_points';
# @param valid_points: a batch of query points, Tensor(v_mask_num, mask_pts_num, 3);
# @param scene_points: Tensor(v_mask_num, mask_scene_pts_num, 3);
# @param lengths_1: Tensor(v_mask_num, );
# @param lengths_2: Tensor(v_mask_num, );
# @param radius: query radius, metrics: m;
#-@return: neighbor point_ids in scene_points for each mask pointcloud, Tensor(v_mask_num, mask_pts_num, K)
def query_neighbors_mt(query_points, scene_points, lengths_1, lengths_2, k=20, radius=0.1):
    _, neighbor_in_scene_pcld, _ = ball_query(query_points, scene_points, lengths_1, lengths_2, K=k, radius=radius)
    return neighbor_in_scene_pcld


# @brief:
# @param pts:
# @param pose:
#-@return
def pts_transform(pts, pose):
    pts_new = torch.sum(pts[..., None, :] * pose[None, :3, :3], dim=-1) + pose[:3, 3][None, ...]
    return pts_new


# @brief:
# @param pts: points needed to be transformed, Tensor(N, 3);
# @param poses: all poses to apply transformations respectively, Tensor(K, 4, 4);
#-@return: Tensor(K, N, 3).
def pts_transform_multi_poses(pts, poses):
    pts_new = torch.sum(pts[None, :, None, :] * poses[:, None, :3, :3], dim=-1)  # Tensor(K, N, 3)
    pts_new = pts_new + poses[:, None, :3, 3]
    return pts_new


# @brief:
# @param pts_xyz: Tensor(n, 3)
# @param xyz_min: Tensor(3, )
# @param xyz_max: Tensor(3, )
def pts_in_bbox(pts_xyz, xyz_min, xyz_max):
    in_range = (pts_xyz >= xyz_min[None, ...]) & (pts_xyz <= xyz_max[None, ...])
    pts_in_range = torch.all(in_range, dim=-1)
    return pts_in_range


# @brief:
# @param tensor_input: input tensor with redundant elements, Tensor(m, );
#-@return uniq_tensor: output tensor without redundant elements, Tensor(n, );
#-@return output_indices: for each element in output tensor, what's its raw index in input tensor, Tensor(n, ).
def tensor_unique(tensor_input):
    uniq_tensor, element_indices = torch.unique(tensor_input, return_inverse=True)
    output_indices = torch.full((uniq_tensor.shape[0],), fill_value=-1, dtype=torch.int64)
    for i in range(element_indices.shape[0]):
        if output_indices[element_indices[i]] == -1:
            output_indices[element_indices[i]] = i
    return uniq_tensor, output_indices


# @brief: for a given 2D bool matrix, return indices of non-zeros elements in each row;
# @param mask_tensor: Tensor(m, n), dtype=bool;
#-@return: Tensor(m, k), dtype=int64.
def nonzero_2d(mask_tensor):
    non_zero_idx_rows = []
    for i in range(mask_tensor.shape[0]):
        non_zero_idx_rows.append( torch.where(mask_tensor[i])[0] )
    row_valid_indices = pad_sequence(non_zero_idx_rows, batch_first=True, padding_value=-1)
    return row_valid_indices


# @brief: for each element in array1, find its index in array2 (array2 has no repeated elements);
# @param array1: Tensor(m, );
# @param array2: Tensor(n, );
#-@return: index in array2 of each array1's element (-1 means this element doesn't appear in array2), Tensor(m, ).
def get_array_indices(array1, array2):
    indices_mapping_list = []
    for i in range(array1.shape[0]):
        element_value = array1[i]
        indices_in_array2 = torch.where(array2 == element_value)[0]
        if indices_in_array2.shape[0] > 0:
            indices_mapping_list.append(indices_in_array2[0].item())
        else:
            indices_mapping_list.append(-1)
    indices_mapping = torch.tensor(indices_mapping_list, dtype=torch.int64)
    return indices_mapping


def transform_points(params, pose_w2c, gaussians_grad=False):
    # Step 1: get centers and norm Rots of Gaussians in World Frame
    if gaussians_grad:
        pts = params['means3D']
    else:
        pts = params['means3D'].detach()

    # Step 2: Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (pose_w2c @ pts4.T).T[:, :3]

    return transformed_pts


# @brief:
# @param pts_3D: means of all GS in Camera CS, Tensor(n, 3);
#-@return: Tensor(n, 3).
def get_depth_and_silhouette(pts_3D):
    # depth of each gaussian center in Camera Frame
    pts_in_cam = torch.cat( (pts_3D, torch.ones_like(pts_3D[:, :1]) ), dim=-1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1)  # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z)  # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)

    return depth_silhouette


# @brief: 组装3D GS在uncertainty rendering时需要用到的Tensor(因为3D GS需要render的物理量必须是3通道的);
# @param pts_ucty: Tensor(n, );
#-@return: Tensor(n, 3).
def get_uncertainty(pts_uncertainty):
    pts_ucty_render = torch.zeros((pts_uncertainty.shape[0], 3)).cuda().float()
    pts_ucty_render[:, 0] = pts_uncertainty
    pts_ucty_render[:, 1] = 1.0
    pts_ucty_render[:, 2] = torch.square(pts_uncertainty)
    return pts_ucty_render


# @brief: project all points (in reconstructed pointcloud) from camera coordinates to pixel coordinates;
# @param pts: camera coordinates of all points in reconstructed pointcloud, Tensor(point_size, 3);
# @param cam_intr: Tensor(3, 3);
#-@return projected_pts: Tensor(point_size, 2);
#-@return front_mask: whether each projected points in front of the camera plane, Tensor(point_size, ).
def compute_projected_pts(pts, cam_intr, z_threshold=0.1):
    homo_pixel_coords = cam_intr @ pts.T
    homo_pixel_coords = homo_pixel_coords.T

    z = homo_pixel_coords[:, -1]  # Tensor(point_size, )
    front_mask = (z > z_threshold)
    z = torch.where(z > 0, z, torch.zeros_like(z) * 1e-5)

    projected_pts = homo_pixel_coords[:, :2] / z[..., None]
    projected_pts = torch.round(projected_pts).to(torch.int64)
    return projected_pts, front_mask


# @param pts_batch: camera coordinates of all points in reconstructed pointcloud, Tensor(K, pts_num, 3);
# @param cam_intr: Tensor(3, 3);
#-@return projected_pts_multi: Tensor(K, pts_num, 2);
#-@return front_mask_multi: Tensor(K, pts_num).
def compute_projected_pts_multi(pts_batch, cam_intr, z_threshold=0.1):
    batch_num, batch_size = pts_batch.shape[0], pts_batch.shape[1]
    pts_a_batch = torch.flatten(pts_batch, end_dim=1)  # Tensor(K * pts_num, 3)
    projected_pts, front_mask = compute_projected_pts(pts_a_batch, cam_intr, z_threshold=z_threshold)
    projected_pts_multi = projected_pts.reshape((batch_num, batch_size, -1))
    front_masks = front_mask.reshape((batch_num, batch_size))
    return projected_pts_multi, front_masks


# @brief: 计算已重建部分点云在当前帧下visible的点, Tensor(pt_num, ), dtype=bool;
# @param pts: Tensor(pt_num, 3);
# @param projected_pts: corresponding pixel coordinates of each point in 3D pointcloud, Tensor(pt_num, 2);
# @param depth_im: Tensor(h, w);
# @param front_mask: Tensor(pt_num, );
#-@return: Tensor(pt_num, ), dtype=bool.
def compute_visibility_mask(pts, projected_pts, depth_im, front_mask, depth_dist_near=0., depth_thresh=0.025):
    im_h, im_w = depth_im.shape
    x, y = projected_pts[:, 0], projected_pts[:, 1]

    # mask1: whether each projected point in image range, and z_value > 0, Tensor(point_size, )
    flag1_1 = (x < 0) | (x >= im_w)
    flag1_2 = (y < 0) | (y >= im_h)
    flag1 = ~(flag1_1 | flag1_2)
    flag1 = flag1 & front_mask

    # mask2: whether each projected point has valid depth
    flag2 = torch.zeros_like(flag1, dtype=torch.bool)
    x_valid, y_valid = x[flag1], y[flag1]
    depth_values = depth_im[y_valid, x_valid]
    flag_depth = (depth_values > depth_dist_near)
    flag2[flag1] = flag_depth

    # mask3: whether the difference of computed depth and GT depth of each projected point is less than threshold
    z = pts[:, 2]
    z_valid = z[flag1]
    flag3 = torch.zeros_like(flag1, dtype=torch.bool)
    flag_depth_dif = ( torch.abs(z_valid - depth_values) < depth_thresh )
    flag3[flag1] = flag_depth_dif

    visibility_mask = (flag1 & flag2 & flag3)
    return visibility_mask


# @brief: 计算已重建部分点云在给定帧下visible的点, Tensor(pt_num, ), dtype=bool
# @param pts: Tensor(pt_num, 3)
# @param projected_pts: corresponding pixel coordinates of each point in 3D pointcloud, Tensor(pt_num, 2)
# @param depth_img: Tensor(h, w);
# @param seg_img: Tensor(h, w);
# @param front_mask: Tensor(pt_num, ), dtype=bool;
#-@return visibility_mask: Tensor(pt_num, ), dtype=bool;
#-@return projected_pts_mask_id: for invisible projected points, its corresponding mask_id == 0, Tensor(pt_num, ), dtype=uint8;
def compute_visibility_mask_ids(pts, projected_pts, depth_img, seg_img, front_mask, depth_dist_near=0., depth_thresh=0.025):
    im_h, im_w = depth_img.shape
    x, y = projected_pts[:, 0], projected_pts[:, 1]  # Tensor(pt_num) / Tensor(K, pt_num)

    # Step 1: compute visibility of each projected point
    # mask1: whether each projected point lies in image range, and has z_value > 0, Tensor(point_size, )
    flag1_1 = (x < 0) | (x >= im_w)
    flag1_2 = (y < 0) | (y >= im_h)
    flag1 = ~(flag1_1 | flag1_2)
    flag1 = flag1 & front_mask

    # mask2: whether each projected point has valid depth
    flag2 = torch.zeros_like(flag1, dtype=torch.bool)
    x_valid, y_valid = x[flag1], y[flag1]
    depth_values = depth_img[y_valid, x_valid]
    flag_depth = (depth_values > depth_dist_near)
    flag2[flag1] = flag_depth

    # mask3: whether the difference of computed depth and GT depth of each projected point is less than threshold (lies in front of the surface of corresponding pixel)
    z = pts[:, 2]
    z_valid = z[flag1]
    flag3 = torch.zeros_like(flag1, dtype=torch.bool)
    flag_depth_dif = (torch.abs(z_valid - depth_values) < depth_thresh)
    flag3[flag1] = flag_depth_dif

    visibility_mask = (flag1 & flag2 & flag3)

    # Step 2: compute corresponding mask_id of each projected points
    projected_pts_mask_id = torch.zeros_like(flag1, dtype=torch.uint8)  # Tensor(pts_num, )
    seg_values = seg_img[y_valid, x_valid]
    valid_id_mask = (seg_values > 0) & flag_depth
    seg_values_valid = torch.where(valid_id_mask, seg_values, torch.zeros_like(seg_values))
    projected_pts_mask_id[flag1] = seg_values_valid

    return visibility_mask, projected_pts_mask_id


# @brief: 计算已重建部分点云在K个给定帧下各自visible的点, Tensor(pt_num, ), dtype=bool
# @param pts: Tensor(K, pt_num, 3)
# @param projected_pts: corresponding pixel coordinates of each point in 3D pointcloud, Tensor(K, pt_num, 2)
# @param depth_imgs: Tensor(K, h, w)
# @param front_masks: Tensor(K, pt_num);
#-@return: Tensor(K, pt_num), dtype=bool
def compute_visibility_masks(pts, projected_pts, depth_imgs, front_masks, depth_dist_near=0., depth_thresh=0.025):
    kf_num, num_gs = pts.shape[0], pts.shape[1]
    im_h, im_w = depth_imgs.shape[1], depth_imgs.shape[2]
    x, y = projected_pts[..., 0], projected_pts[..., 1]  # Tensor(K, pt_num) / Tensor(K, pt_num)
    kf_indices = torch.arange(0, kf_num)[..., None].tile((1, num_gs)).to(projected_pts)  # Tensor(K, pt_num)

    # mask1: whether each projected point in image range, and z_value > 0, Tensor(point_size, )
    flag1_1 = (x < 0) | (x >= im_w)
    flag1_2 = (y < 0) | (y >= im_h)
    flag1 = ~(flag1_1 | flag1_2)
    flag1 = flag1 & front_masks  # Tensor(K, pt_num), dtype=bool

    # mask2: whether each projected point has valid depth
    flag2 = torch.zeros_like(flag1, dtype=torch.bool)
    x_valid, y_valid = x[flag1], y[flag1]
    kf_indices_valid = kf_indices[flag1]
    depth_values = depth_imgs[kf_indices_valid, y_valid, x_valid]
    flag_depth = (depth_values > depth_dist_near)
    flag2[flag1] = flag_depth  # Tensor(K, pt_num), dtype=bool

    # mask3: whether the difference of computed depth and GT depth of each projected point is less than threshold
    z = pts[..., 2]  # Tensor(K, pt_num)
    z_valid = z[flag1]
    flag3 = torch.zeros_like(flag1, dtype=torch.bool)
    flag_depth_dif = (torch.abs(z_valid - depth_values) < depth_thresh)
    flag3[flag1] = flag_depth_dif  # Tensor(K, pt_num), dtype=bool

    visibility_mask = (flag1 & flag2 & flag3)
    return visibility_mask


# @brief: 计算已重建部分点云在K个给定帧下各自visible的点, Tensor(pt_num, ), dtype=bool
# @param pts: Tensor(K, pt_num, 3)
# @param projected_pts: corresponding pixel coordinates of each point in 3D pointcloud, Tensor(K, pt_num, 2)
# @param depth_imgs: Tensor(K, h, w);
# @param seg_imgs: Tensor(K, h, w);
# @param front_masks: Tensor(K, pt_num);
#-@return visibility_mask: Tensor(K, pt_num), dtype=bool;
#-@return projected_pts_mask_id: Tensor(K, pt_num), dtype=uint8;
def compute_visibility_masks_ids(pts, projected_pts, depth_imgs, seg_imgs, front_masks, depth_dist_near=0., depth_thresh=0.025):
    kf_num, num_gs = pts.shape[0], pts.shape[1]
    im_h, im_w = depth_imgs.shape[1], depth_imgs.shape[2]
    x, y = projected_pts[..., 0], projected_pts[..., 1]  # Tensor(K, pt_num) / Tensor(K, pt_num)
    kf_indices = torch.arange(0, kf_num)[..., None].tile((1, num_gs)).to(projected_pts)  # Tensor(K, pt_num)

    # Step 1: compute visibility of each projected point
    # mask1: whether each projected point in image range, and z_value > 0, Tensor(point_size, )
    flag1_1 = (x < 0) | (x >= im_w)
    flag1_2 = (y < 0) | (y >= im_h)
    flag1 = ~(flag1_1 | flag1_2)
    flag1 = flag1 & front_masks  # Tensor(K, pt_num), dtype=bool

    # mask2: whether each projected point has valid depth
    flag2 = torch.zeros_like(flag1, dtype=torch.bool)  # Tensor(K, pts_num), dtype=bool
    x_valid, y_valid = x[flag1], y[flag1]
    kf_indices_valid = kf_indices[flag1]
    depth_values = depth_imgs[kf_indices_valid, y_valid, x_valid]
    flag_depth = (depth_values > depth_dist_near)
    flag2[flag1] = flag_depth  # Tensor(K, pt_num), dtype=bool

    # mask3: whether the difference of computed depth and GT depth of each projected point is less than threshold
    z = pts[..., 2]  # Tensor(K, pt_num)
    z_valid = z[flag1]
    flag3 = torch.zeros_like(flag1, dtype=torch.bool)
    flag_depth_dif = (torch.abs(z_valid - depth_values) < depth_thresh)
    flag3[flag1] = flag_depth_dif  # Tensor(K, pt_num), dtype=bool

    visibility_mask = (flag1 & flag2 & flag3)

    # Step 2: compute corresponding mask_id of each projected points
    projected_pts_mask_id = torch.zeros_like(flag1, dtype=torch.uint8)  # Tensor(K, pts_num)
    seg_values = seg_imgs[kf_indices_valid, y_valid, x_valid]
    valid_id_mask = (seg_values > 0) & flag_depth
    seg_values_valid = torch.where(valid_id_mask, seg_values, torch.zeros_like(seg_values))
    projected_pts_mask_id[flag1] = seg_values_valid

    return visibility_mask, projected_pts_mask_id


# @brief: 计算点云中的每个visible的点在当前帧的各2D instance mask下的visibility情况;
# @param projected_pts: 点云中的各点投影到当前帧像素平面上的像素坐标, Tensor(pts_num, 2);
# @param visibility_mask: 点云中的各点在当前帧中的visible mask, Tensor(pts_num, ), dtype=bool;
# @param pred_masks: 当前帧的各2D instance的mask, Tensor(v_instance_num, H, W), dtype=bool;
# @param score_map: 当前帧中各pixel在2D instance上的score, Tensor(v_instance_num, H, W), dtype=float32;
#-@return: 点云中的每个点在各个instance mask下的visibility mask, Tensor(v_instance_num, pts_num), dtype=bool.
def compute_visible_masked_pts(projected_pts, visibility_mask, pred_masks, score_map):
    instance_num = pred_masks.shape[0]
    x, y = projected_pts[:, 0], projected_pts[:, 1]  # point_Id为i的点的像素坐标
    x[~visibility_mask] = 0
    y[~visibility_mask] = 0

    visible_mask_m = visibility_mask[None, ...].repeat((instance_num, 1))  # Tensor(v_instance_num, pts_num), dtype=bool;

    pred_pts = pred_masks[:, y, x]  # 各个3D点投影后的像素坐标是否在每个2D valid instance的mask中
    masked_pts = torch.where(visible_mask_m, pred_pts, torch.zeros_like(visible_mask_m))

    score_pts = score_map[:, y, x]  # 每个3D点在各个valid instance下的score, Tensor(v_instance_num, pts_num)
    masked_pts_score = torch.where(visible_mask_m, score_pts, torch.zeros_like(score_pts))  # Tensor(v_instance_num, pts_num)
    masked_pts_score = masked_pts_score * masked_pts.to(masked_pts_score)
    return masked_pts, masked_pts_score


def compute_pixel_mask(img_h, img_w, downsample_ratio_h, downsample_ratio_w, device="cuda:0"):
    num_h = int(img_h // downsample_ratio_h)
    num_w = int(img_w // downsample_ratio_w)
    rows, cols = sample_pixels_uniformly(img_h, img_w, num_h, num_w)
    pixel_indices = pixel_rc_to_indices(img_h, img_w, rows, cols)

    pixel_mask = torch.zeros((img_h * img_w,), dtype=torch.bool, device=device)
    pixel_mask[pixel_indices] = True
    return pixel_mask


# @param pt_cloud: Tensor(pts_num, 3)
# @param masked_pts: Tensor(v_instance_num, pts_num)
#-@return: Tensor(v_instance_num, 6)
def get_instance_bbox(pt_cloud, masked_pts):
    v_instance_num = masked_pts.shape[0]
    instance_min_max = torch.zeros((v_instance_num, 6), dtype=torch.float32)
    for i in range(v_instance_num):
        instance_mask = masked_pts[i]
        instance_pts = pt_cloud[instance_mask, :]  # Tensor(k, 3)
        xyz_min = torch.min(instance_pts, 0)[0]
        xyz_max = torch.max(instance_pts, 0)[0]
        instance_min_max[i, :3] = xyz_min
        instance_min_max[i, 3:] = xyz_max
    return instance_min_max


# @brief: compute union of 2 given bboxes
# @param xyz_min_1: [x_min_1, y_min_1, z_min_1], Tensor(3, );
# @param xyz_max_1: [x_max_1, y_max_1, z_max_1], Tensor(3, );
# @param xyz_min_2: [x_min_2, y_min_2, z_min_2], Tensor(3, );
# @param xyz_max_2: [x_max_2, y_max_2, z_max_2], Tensor(3, );
#-@return xyz_min: [x_min_final, y_min_final, z_min_final], Tensor(3, );
#-@return xyz_max: [x_max_final, y_max_final, z_max_final], Tensor(3, ).
def bbox_merge(xyz_min_1, xyz_max_1, xyz_min_2, xyz_max_2):
    xyz_min12 = torch.stack([xyz_min_1, xyz_min_2], dim=1)  # Tensor(3, 2)
    xyz_max12 = torch.stack([xyz_max_1, xyz_max_2], dim=1)  # Tensor(3, 2)
    xyz_min = xyz_min12.min(dim=-1)[0]
    xyz_max = xyz_max12.max(dim=-1)[0]
    return xyz_min, xyz_max


# @brief: compute union of 2 given bboxes
# @param bbox1: [x_min_1, y_min_1, z_min_1, x_max_1, y_max_1, z_max_1];
# @param bbox2: [x_min_2, y_min_2, z_min_2, x_max_2, y_max_2, z_max_2];
#-@return
def bbox_merge2(bbox1, bbox2):
    xyz_min12 = torch.stack([bbox1[:3], bbox2[:3]], dim=1)
    xyz_max12 = torch.stack([bbox1[3:], bbox2[3:]], dim=1)
    xyz_min = xyz_min12.min(dim=-1)[0]
    xyz_max = xyz_max12.max(dim=-1)[0]
    bbox = torch.cat([xyz_min, xyz_max], dim=0)
    return bbox


# @brief: get points lying within the given bounding box;
# @param points: Tensor(n, 3);
# @param bbox: open3d.geometry.OrientedBoundingBox obj;
#-@return valid_pts: points lying within the bounding box, Tensor(n', 3);
#-@return valid_pts_indices: indices of points lying within the bounding box, Tensor(n', ).
def get_pts_within_bbox(points, bbox):
    pts_vector = o3d.utility.Vector3dVector(points.cpu().numpy())
    valid_pts_indices = bbox.get_point_indices_within_bounding_box(pts_vector)
    valid_pts_indices = torch.tensor(valid_pts_indices)
    valid_pts = points[valid_pts_indices]
    return valid_pts, valid_pts_indices


# @brief: for a given pointcloud, filter out points with any NaN value
def pc_filtering(pc, has_color=True):
    pc_points = np.asarray(pc.points)  # ndarray(N, 3), dtype=float64
    if has_color:
        pc_colors = np.asarray(pc.colors)  # ndarray(N, 3), dtype=float64

    invalid_row_mask = np.isnan(pc_points).sum(axis=-1).astype("bool")
    valid_row_mask = ~invalid_row_mask
    pc_points = pc_points[valid_row_mask]
    pc_filtered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_points))

    if has_color:
        pc_colors = pc_colors[valid_row_mask]
        pc_filtered.colors = o3d.utility.Vector3dVector(pc_colors)
    return pc_filtered


def pc_align(pc1, pc2, threshold=0.05):
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pc1, pc2, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint() )
    return reg_p2p.transformation


# @param x: Tensor(M, d);
# @param x: Tensor(N, d);
#-@return: Tensor(M, N).
def pairwise_cosine_similarity(x, y):
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    return x_norm @ y_norm.T


# @brief: merge 3D instances by adjacent matrix;
# @param adj_matrix: Tensor(N, N), dtype=bool;
#-@return 合并后的各instance与合并前的各instance的对照关系, list of list.
def find_connected_components(adj_matrix):
    if torch.is_tensor(adj_matrix):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "adjacency matrix should be a square matrix"

    N = adj_matrix.shape[0]  # number of existing instances
    clusters = []
    visited = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        if visited[i]:
            continue
        cluster = []
        queue = deque([i])
        visited[i] = True
        while queue:
            j = queue.popleft()
            cluster.append(j)
            for k in np.nonzero(adj_matrix[j])[0]:
                if not visited[k]:
                    queue.append(k)
                    visited[k] = True
        clusters.append(cluster)
    return clusters


# @brief: 给定一组3D点，保存得到它们的点云
def save_pts_ply(pts, save_path=None, pc_color=[1., 0., 0.]):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.estimate_normals()
    pc.paint_uniform_color(np.array(pc_color).astype("float64"))

    if save_path is not None:
        o3d.io.write_point_cloud(save_path, pc)
    return pc


# @brief: 给定一张seg image, 将它转换成RGB图并保存
# @param seg_img: Tensor(H, W), dtype=uint8
def save_seg_image(seg_img, save_path=None):
    mask_values = torch.unique(seg_img)
    mask_values, _ = torch.sort(mask_values)

    rgb_value_list = get_new_pallete(mask_values.shape[0])  # list of int (3 * label_num)
    mask_color_image = np.zeros((seg_img.shape[0], seg_img.shape[1], 3), dtype=np.uint8)  # 3通道RGB mask

    for i in range(mask_values.shape[0]):
        mask_value = mask_values[i]
        if mask_value == 0:
            continue
        value_mask = (seg_img == mask_value)
        num_pixels = torch.sum(seg_img[value_mask])
        if num_pixels < 400:
            # ignore small masks
            continue

        # 绘制mask_id map
        # 给每个mask_id对应的mask上不同的的颜色
        rgb_value = [rgb_value_list[i * 3], rgb_value_list[i * 3 + 1], rgb_value_list[i * 3 + 2]]
        mask_color_image[value_mask.cpu().numpy()] = rgb_value

    if save_path is not None:
        cv2.imwrite(save_path, mask_color_image)
    return mask_color_image


def save_seg_image_select_mask(seg_img, given_mask_id, save_path=None):
    mask_values = torch.unique(seg_img)
    mask_values, _ = torch.sort(mask_values)

    rgb_value_list = get_new_pallete(mask_values.shape[0])  # list of int (3 * label_num)
    mask_color_image = np.zeros((seg_img.shape[0], seg_img.shape[1], 3), dtype=np.uint8)  # 3通道RGB mask

    for i in range(mask_values.shape[0]):
        mask_value = mask_values[i]
        if mask_value == 0 or mask_value != given_mask_id:
            continue
        value_mask = (seg_img == mask_value)
        num_pixels = torch.sum(seg_img[value_mask])
        if num_pixels < 400:
            # ignore small masks
            continue

        # 绘制mask_id map
        # 给每个mask_id对应的mask上不同的的颜色
        rgb_value = [rgb_value_list[i * 3], rgb_value_list[i * 3 + 1], rgb_value_list[i * 3 + 2]]
        mask_color_image[value_mask.cpu().numpy()] = rgb_value

    if save_path is not None:
        cv2.imwrite(save_path, mask_color_image)
    return mask_color_image


def save_seg_image_select_mask_ply(depth_img, seg_img, given_mask_id, intrinsic=None, save_path=None):
    mask_values = torch.unique(seg_img)
    mask_values, _ = torch.sort(mask_values)

    depth_img = depth_img.cpu().numpy()
    mask_depth_img = np.zeros_like(depth_img)

    for i in range(mask_values.shape[0]):
        mask_value = mask_values[i]
        if mask_value == 0 or mask_value != given_mask_id:
            continue
        value_mask = (seg_img == mask_value)
        num_pixels = torch.sum(seg_img[value_mask])
        if num_pixels < 400:
            # ignore small masks
            continue

        # 绘制mask_id map
        # 给每个mask_id对应的mask上不同的的颜色
        mask_depth_img[value_mask.cpu().numpy()] = depth_img[value_mask.cpu().numpy()]

    if intrinsic is not None and save_path is not None:
        final_pc, _, _ = depth2pcl(mask_depth_img, intrinsic.cpu().numpy())
        save_pts_ply(final_pc, save_path, pc_color=[0., 1., 0.])
    return mask_depth_img


#-@return final_pc: 保留下来的点, ndarray(N, 3)
#-@return ori_point_cloud: ndarray(H, W, 3)
#-@return mask: 哪些点被保留了下来的mask
def depth2pcl(depth_img, intrinsic, depth_scale=1.):
    '''
    深度图转点云数据
    :param depth_img: 深度图
    :param intrinsic: 内参矩阵
    :return: point_cloud  np.array(N, 3)
    '''
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    H, W = depth_img.shape
    h,w = np.mgrid[0:H, 0:W]
    z = depth_img / depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy

    point_cloud = np.stack([x, y, z], -1).reshape(-1, 3)
    index = (point_cloud[:, -1] == 0)
    ori_point_cloud = np.stack([x, y, z], -1)
    final_pc = np.delete(point_cloud, index, axis=0)
    mask = ~index
    return final_pc, ori_point_cloud, mask


def save_projection_image(H, W, projected_pixel_uv, pixel_mask, save_path=None, color=[255, 0, 0]):
    mask_color_image = np.zeros((H, W, 3), dtype=np.uint8)  # 3通道RGB mask
    valid_pixels = projected_pixel_uv[pixel_mask]
    for v_pixel in valid_pixels:
        col_id, row_id = v_pixel[0].item(), v_pixel[1].item()
        mask_color_image[row_id, col_id] = color

    if save_path is not None:
        cv2.imwrite(save_path, mask_color_image)
    return mask_color_image

