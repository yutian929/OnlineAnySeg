import cv2
from collections import defaultdict
import open3d as o3d
import networkx as nx
import math
import numpy as np
import torch
import torch.nn.functional as F


def create_lower_triangular_matrix(n, device="cuda:0", dtype=torch.bool):
    matrix = torch.ones((n, n))  # 创建一个全1的n阶方阵
    matrix = torch.tril(matrix, -1)  # 将对角线及以上的元素设置为0
    matrix = matrix.to(device)
    matrix = matrix.type(dtype)
    return matrix


# @brief: for a given square matrix, remove specific rows+cols
# @param matrix: input matrix;
# @param rows_to_remove: row_ids(also col_ids) need to be removed;
#-@return: remaining matrix.
def remove_rows_and_cols(matrix, rows_to_remove):
    rows_to_remove = torch.tensor(rows_to_remove, dtype=torch.long)

    # Create a mask for the rows and columns to keep
    mask = torch.ones(matrix.size(0), dtype=torch.bool)
    mask[rows_to_remove] = False
    reduced_matrix = matrix[mask][:, mask]  # get row and cols to keep
    return reduced_matrix

# @brief: keep specific rows for given matrix, setting other rows to zeros
def remain_rows(matrix, rows_to_keep):
    mask = torch.zeros_like(matrix)
    mask[rows_to_keep, :] = 1
    result = matrix * mask
    return result

def set_diagonal_to_zero(matrix):
    identity_matrix = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)  # 生成一个与输入矩阵相同大小的单位矩阵
    mask = 1 - identity_matrix  # 将单位矩阵的对角线元素设置为 0，其余元素设置为 1
    result_matrix = matrix * mask  # 使用掩码矩阵将输入矩阵的对角线元素置为 0
    return result_matrix


def set_row_and_column_zero(matrix, target_row, target_col):
    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("Input matrix must be a square matrix.")
    modified_matrix = matrix.clone()  # 创建一个副本以避免修改原始矩阵（如果需要）
    modified_matrix[target_row, :] = 0  # 将指定的行置为0
    modified_matrix[:, target_col] = 0  # 将指定的列置为0
    return modified_matrix

def query_values_from_keys(input_dict, query_keys, device="cuda:0"):
    sorted_dict = dict( sorted(input_dict.items()) )
    keys = torch.tensor(list(sorted_dict.keys()), device=device)
    values = torch.tensor(list(sorted_dict.values()), device=device)
    query_indices = torch.bucketize(query_keys, keys)
    query_values = values[query_indices]
    return query_values


def get_intrinsics(intrinsic_mat, h, w):
    intrinsic_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_cam_parameters.set_intrinsics(w, h, intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2])
    return intrinsic_cam_parameters

def has_duplicate_coords(coords):
    if coords.size(0) != torch.unique(coords, dim=0).size(0):
        return True
    return False

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
#-@return mean3_sq_dist: Tensor(valid_pts_num, ).
def get_pointcloud(color, depth, intrinsics, pose_c2w, transform_pts=True, mask=None, gs_ratio=1.5):
    width, height = color.shape[1], color.shape[0]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), torch.arange(height).cuda().float(), indexing="xy")
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth.reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)  # Tensor(h * w, 3)
    if transform_pts:  # default
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        pts = (pose_c2w @ pts4.T).T[:, :3]  # point world coordinates, Tensor(h * w, 3)
    else:
        pts = pts_cam
    cols = color.reshape(-1, 3)  # Tensor(h, w, 3) -> Tensor(h * w, 3)

    if mask is not None:
        depth_z = depth_z[mask]
        pts = pts[mask]
        cols = cols[mask]

    # Compute mean squared distance for initializing the scale of the Gaussians
    # Projective Geometry (this is fast, farther -> larger radius)
    scale_gaussian = gs_ratio * ( depth_z / ((FX + FY) / 2) )  # Tensor(h * w)
    mean3_sq_dist = scale_gaussian ** 2  # Tensor(h * w)

    point_cld = torch.cat((pts, cols), -1)  # Tensor(h * w, 6), [x, y, z, R, G, B]
    return point_cld, mean3_sq_dist


# @brief: get Gaussian poincloud for a given RGB-D frame
# @param depth: Tensor(h, w);
# @param intrinsics: Tensor(3, 3);
# @param pose_c2w: Tensor(4, 4);
# @param transform_pts: whether transform point from Camera coordinates to World coordinates, bool;
# @param mask: Tensor(h * w), dtype=bool;
#-@return point_cld: [x, y, z], Tensor(valid_pts_num, 3).
def get_pointcloud_xyz(depth, intrinsics, pose_c2w, transform_pts=True, mask=None):
    width, height = depth.shape[1], depth.shape[0]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().double(), torch.arange(height).cuda().double(), indexing="xy")
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth.reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)  # Tensor(h * w, 3)
    # pts_cam = torch.stack([x, y, z], dim=-1).reshape(-1, 3)  # Tensor(h * w, 3)
    if transform_pts:  # default
        pix_ones = torch.ones(height * width, 1).cuda().double()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        pts = (pose_c2w.double() @ pts4.T).T[:, :3]  # point world coordinates, Tensor(h * w, 3)
    else:
        pts = pts_cam

    if mask is not None:
        pts = pts[mask]
    return pts


# @brief: for N sets, get their union set, and count each unique element;
# @param set_list: list of set;
#-@return union_tensor: sorted union set Tensor(n, );
#-@return counts_tensor: count of each element in union set, Tensor(n, ).
def merge_sets_and_count(set_list, device="cuda:0"):
    element_counts = defaultdict(int)  # Create a dictionary to store the count of each element
    union_set = set()  # Create a set to store the union of all sets

    # Iterate through each set
    for s in set_list:
        if s is None:
            continue
        union_set.update(s)  # Update the union set

        # Update the counts for each element in the current set
        for elem in s:
            element_counts[elem] += 1

    sorted_union_list = sorted( list(union_set) )  # Convert union set to a sorted list
    union_tensor = torch.tensor(sorted_union_list, device=device)

    counts_tensor = torch.tensor([element_counts[elem] for elem in sorted_union_list], device=device)  # counts of each element in the union set
    return union_tensor, counts_tensor


# @brief: 从矩阵中提取指定的行和列
# @param matrix: 输入的 n x n 矩阵 (tensor)
# @param indices: 要提取的行和列的索引 (list or tensor)
#-@return: 提取出的 m x m 子矩阵 (tensor)
def extract_rows_and_cols(matrix, indices):
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, dtype=torch.long)  # 将 indices 转换为张量类型
    extracted_matrix = matrix[indices][:, indices]  # 提取指定的行和列
    return extracted_matrix

# @brief: 保留每一行中最大的元素，其余元素置为0;
# @param matrix:
def retain_max_per_row(matrix):
    max_values, max_indices = torch.max(matrix, dim=-1)  # 获取每行的最大值及其索引
    result = torch.zeros_like(matrix)  # 创建一个与输入矩阵相同形状的零矩阵
    result[torch.arange(matrix.shape[0]), max_indices] = max_values  # 使用高级索引，将每行最大值的位置置为原始矩阵中的值
    return result

# @brief: 保留每一列中最大的元素，其余元素置为0;
# @param matrix:
def retain_max_per_column(matrix):
    max_values, max_indices = torch.max(matrix, dim=0)  # 获取每列的最大值及其索引
    result = torch.zeros_like(matrix)  # 创建一个与输入矩阵相同形状的零矩阵
    result[max_indices, torch.arange(matrix.shape[1])] = max_values  # 使用高级索引，将每列最大值的位置置为原始矩阵中的值
    return result


# @brief: Retain values at specified rows and columns, set the rest to 0;
# @param matrix: The input n * n matrix;
# @param indices: The indices of rows and columns to retain, Tensor(m, );
#-@return: The modified matrix with specified rows and columns retained.
def mask_matrix_rows_and_cols(matrix, indices, default_value=0.):
    if default_value == 0:
        masked_matrix = torch.zeros_like(matrix)
    else:
        masked_matrix = default_value * torch.ones_like(matrix)
    masked_matrix[indices[:, None], indices] = matrix[indices[:, None], indices]
    return masked_matrix


# @brief: 将B中的各元素值赋到A的指定的对应行和列的元素中
def assign_elements_2d(A, B, row_indices, col_indices):
    if len(row_indices) == 0 or len(col_indices) == 0:
        return A
    # 获取行列索引的笛卡尔积
    row_indices = torch.tensor(row_indices).unsqueeze(1)  # 扩展为列向量
    col_indices = torch.tensor(col_indices).unsqueeze(0)  # 扩展为行向量

    # 通过广播将行列索引组合
    A[row_indices, col_indices] = B
    return A

def update_dict_key(dict_to_modify, old_key, new_key):
    if not old_key in dict_to_modify:
        return False
    else:
        value = dict_to_modify[old_key]
        del dict_to_modify[old_key]
        dict_to_modify[new_key] = value
        return True

def delete_dict_keys(dict_to_modify, keys_to_delete):
    for key_to_delete in keys_to_delete:
        if not key_to_delete in dict_to_modify:
            continue
        else:
            del dict_to_modify[key_to_delete]


# @brief: do clustering for each mask using given adjacent matrix (find each connected component);
# @param adj_matrix: Tensor(mask_num, mask_num), dtype=bool;
#-@return: global mask_id of each mask clustering, list of ndarray(n_i).
def do_clustering(adj_matrix):
    adj_matrix = adj_matrix.cpu().numpy()

    G = nx.from_numpy_array(adj_matrix)  # Graph obj
    mask_id_clusters = []
    for component in nx.connected_components(G):
        component_mask_ids = list(component)
        mask_id_clusters.append( np.sort(component_mask_ids) )
    return mask_id_clusters
