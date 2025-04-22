import torch
from pytorch3d.ops import ball_query
import numpy as np
import open3d as o3d
from math import exp


def get_depth_mask(depth_tensor, dist_far=10.):
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


# @brief: given a pointcloud, remove noise points from it (only maximal connected component is kept)
# @param
def denoise(pcd, eps=0.04):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=4)) + 1  # -1 for noise
    mask = np.zeros(len(labels), dtype=bool)
    count = np.bincount(labels)
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
    count = np.bincount(labels)

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
    first_true_mask = (cumsum == 1)

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
    numerator = 1 - torch.exp(-k * (x - min_value))
    denominator = 1 - exp(-k * (max_value - min_value))

    mapped_x = numerator / denominator
    return mapped_x


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
