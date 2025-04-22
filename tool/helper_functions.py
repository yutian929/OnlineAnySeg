import cv2
from collections import defaultdict
import networkx as nx
import numpy as np
import torch


def create_lower_triangular_matrix(n, device="cuda:0", dtype=torch.bool):
    matrix = torch.ones((n, n))
    matrix = torch.tril(matrix, -1)
    matrix = matrix.to(device)
    matrix = matrix.type(dtype)
    return matrix

def set_diagonal_to_zero(matrix):
    identity_matrix = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    mask = 1 - identity_matrix
    result_matrix = matrix * mask
    return result_matrix

def set_row_and_column_zero(matrix, target_row, target_col):
    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("Input matrix must be a square matrix.")
    modified_matrix = matrix.clone()
    modified_matrix[target_row, :] = 0
    modified_matrix[:, target_col] = 0
    return modified_matrix

def query_values_from_keys(input_dict, query_keys, device="cuda:0"):
    sorted_dict = dict( sorted(input_dict.items()) )
    keys = torch.tensor(list(sorted_dict.keys()), device=device)
    values = torch.tensor(list(sorted_dict.values()), device=device)
    query_indices = torch.bucketize(query_keys, keys)
    query_values = values[query_indices]
    return query_values


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

def extract_rows_and_cols(matrix, indices):
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, dtype=torch.long)
    extracted_matrix = matrix[indices][:, indices]
    return extracted_matrix

def retain_max_per_row(matrix):
    max_values, max_indices = torch.max(matrix, dim=-1)
    result = torch.zeros_like(matrix)
    result[torch.arange(matrix.shape[0]), max_indices] = max_values
    return result

def retain_max_per_column(matrix):
    max_values, max_indices = torch.max(matrix, dim=0)
    result = torch.zeros_like(matrix)
    result[max_indices, torch.arange(matrix.shape[1])] = max_values
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


# @brief: assign values of B into specified rows and cols in A
def assign_elements_2d(A, B, row_indices, col_indices):
    if len(row_indices) == 0 or len(col_indices) == 0:
        return A
    row_indices = torch.tensor(row_indices).unsqueeze(1)
    col_indices = torch.tensor(col_indices).unsqueeze(0)
    A[row_indices, col_indices] = B
    return A


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
