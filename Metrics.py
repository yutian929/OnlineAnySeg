import numpy as np
import torch
import torch.nn.functional as F

from tool.geometric_helpers import nonlinear_mapping
from tool.helper_functions import retain_max_per_row, create_lower_triangular_matrix


class Metrics:
    def __init__(self, cfg, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        self.seman_min_value = self.cfg["seg"]["seman_min_value"]


    @staticmethod
    def compute_similarity_matrix(fused_features):
        similarity_matrix = F.cosine_similarity(fused_features.unsqueeze(1), fused_features.unsqueeze(0), dim=-1)
        return similarity_matrix


    # @brief: compute final similarity matrix based on overlap / semantic similarity / geometric similarity;
    # @param c_mat: containing matrix of all existing masks, Tensor(mask_num, mask_num);
    # @param sem_feature_mat: semantic feature similarity matrix of all existing masks, Tensor(mask_num, sem_feat_dim);
    # @param geo_feature_mat: geometric feature similarity matrix of all existing masks, Tensor(mask_num, geo_feat_dim);
    # @param masks_to_merged
    # @param masks_to_receive
    # @param retain_max: whether for each row, only max value is kept;
    @torch.no_grad()
    def compute_final_sim_mat(self, c_mat, sem_feature_mat, geo_feature_mat, masks_to_merged=None, masks_to_receive=None, retain_max=False, use_IoU=True):
        # Step 1: IoU matrix, shape=(n_a, n_e)
        iou_mat = (c_mat + c_mat.T) / 2  # Tensor(c_mask_num, c_mask_num), dtype=float32
        if masks_to_merged is not None and masks_to_receive is not None:
            iou_mat = iou_mat[masks_to_merged][:, masks_to_receive]  # Tensor(n_a, n_e)

        # Step 2: semantic feature similarity matrix, shape=(n_a, n_e)
        sem_feature_mat = sem_feature_mat / (sem_feature_mat.norm(dim=-1, keepdim=True) + 1e-7)  # normalization for extracted visual embeddings
        sem_feature_sim_mat = sem_feature_mat @ sem_feature_mat.T  # Tensor(c_mask_num, c_mask_num), dtype=float32
        if masks_to_merged is not None and masks_to_receive is not None:
            sem_feature_sim_mat = sem_feature_sim_mat[masks_to_merged][:, masks_to_receive]  # Tensor(n_a, n_e)
        sem_feature_sim_mat = nonlinear_mapping(sem_feature_sim_mat, min_value=self.seman_min_value, max_value=1.)  # non-linear mapping for all values, Tensor(n_a, n_e)

        # Step 3: geometric feature similarity matrix, shape=(n_a, n_e)
        geo_feature_mat = geo_feature_mat / (geo_feature_mat.norm(dim=-1, keepdim=True) + 1e-7)  # normalization for extracted visual embeddings
        geo_feature_sim_mat = geo_feature_mat @ geo_feature_mat.T  # Tensor(c_mask_num, c_mask_num), dtype=float32
        if masks_to_merged is not None and masks_to_receive is not None:
            geo_feature_sim_mat = geo_feature_sim_mat[masks_to_merged][:, masks_to_receive]  # Tensor(n_a, n_e)

        # Step 4: compute final similarity matrix, and for each row only the element with max value will be kept
        if use_IoU:
            final_sim_mat = iou_mat + (sem_feature_sim_mat + geo_feature_sim_mat)
        else:
            final_sim_mat = sem_feature_sim_mat + geo_feature_sim_mat

        if retain_max:
            final_sim_mat = retain_max_per_row(final_sim_mat)
        return final_sim_mat, iou_mat, sem_feature_sim_mat, geo_feature_sim_mat






