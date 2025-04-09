import numpy as np
import torch
import torch.nn.functional as F

from tool.geometric_helpers import nonlinear_mapping
from tool.helper_functions import retain_max_per_row, create_lower_triangular_matrix
from Bi_plane_classifier import Bi_Plane_Classifier


class Metrics:
    def __init__(self, cfg, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        self.seman_min_value = self.cfg["seg"]["seman_min_value"]

        ################## members for Bi-plane classifier ##################
        self.total_mask_num = self.cfg["scene"]["mask_num"]  # total mask num
        # self.tracked_id2merged_id = -1 * torch.ones((self.total_mask_num, ), dtype=torch.float32, device=self.device)  # 记录每个tracked mask当前对应的merged_mask_ID (需同步更新)
        self.mapping_frame_Id = -1
        self.raw_ID2new_ID = None

        self.bi_plane = Bi_Plane_Classifier(cfg, device)
        self.trust_area = self.bi_plane.trust_area
        self.hesitate_area = self.bi_plane.hesitate_area
        self.hes_mask_IDs = self.bi_plane.hes_mask_IDs  # Hesitate Area中mask_index到真正的 merged_mask_ID 的映射表 (避免每次mask merging后要更新2个area中的 Mask_pair objs)
    # END init()


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
    #-@return:
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


    ###################################### Bi-Plane mask merging strategy ######################################

    # @brief: judge whether each pair should be merged;
    #-@return: Merging mask matrix, Tensor(mask_num, mask_num), dtype=bool.
    def bi_plane_classify(self, iou_mat, geo_feature_sim_mat, sem_feature_sim_mat):
        positive_mask = self.bi_plane.do_classification(iou_mat, geo_feature_sim_mat, sem_feature_sim_mat)  # mask of whether a pair falls above the classifier plane, Tensor(mask_num, mask_num), dtype=bool;
        trust_mask, _ = self.bi_plane.judge_trust_area(iou_mat, sem_feature_sim_mat, geo_feature_sim_mat)  # mask of whether a pair is in Trust Area, Tensor(mask_num, mask_num), dtype=bool;
        merge_mask = torch.logical_and(positive_mask, trust_mask)
        return merge_mask


    # @brief: Step 1 (Merge前): for all currently existing merged masks, find all candidate mask pairs, and put each pair into either Trust Area or Hesitating Area;
    def find_mask_pairs(self, frame_ID, c_mat, iou_mat, sem_feature_sim_mat, geo_feature_sim_mat, c_thresh=0.1, ut_mask=None):
        mask_num = c_mat.shape[0]
        if ut_mask is None:
            ut_mask = create_lower_triangular_matrix(mask_num)

        # Step 1: find all candidate mask pairs
        candi_pair_mask = ut_mask & (c_mat > c_thresh) & (c_mat.T > c_thresh)
        self.bi_plane.judge_candidate_pairs(frame_ID, iou_mat, sem_feature_sim_mat, geo_feature_sim_mat, candi_pair_mask)


    # @brief: : When each time the IDs of all merged masks are reassigned, the candidate mask paris in Hesitate Area should update its IDs synchronously;
    # @param raw_ID2new_ID: merging & removing 前的各merged_mask_ID到之后的merged_mask_ID的映射表, Tensor(raw_merged_mask_num, );
    def update_mask_pairs_IDs(self, raw_ID2new_ID):
        hes_pair_num = len(self.hesitate_area)
        raw_pair_mask_IDs = self.hes_mask_IDs[:hes_pair_num]  # Tensor(hes_pair_num, 2)
        new_pair_mask_IDs_1 = raw_ID2new_ID[raw_pair_mask_IDs[:, 0]]
        new_pair_mask_IDs_2 = raw_ID2new_ID[raw_pair_mask_IDs[:, 1]]
        new_pair_mask_IDs = torch.stack([new_pair_mask_IDs_1, new_pair_mask_IDs_2], dim=-1)

        self.hes_mask_IDs[:hes_pair_num] = new_pair_mask_IDs


    # @brief: Step 2 (Merge后): 对本轮合并前 Hesitate Area 中的所有mask pairs, 重新计算它们合并后的scores, 并判断哪些需要添加到 Positive/Negative set 中
    def update_classifier(self, frame_ID, raw_ID2new_ID, iou_mat, sem_feature_sim_mat, geo_feature_sim_mat):
        # Step 1: update mask_IDs for tracked mask pairs after merging
        self.update_mask_pairs_IDs(raw_ID2new_ID)

        # Step 2: for all mask pairs in Hesitate Area currently, recompute its score; and find pairs appended into Positive/Negative Set
        self.bi_plane.rejudge_hesitate_pairs(frame_ID, iou_mat, sem_feature_sim_mat, geo_feature_sim_mat)

        # Step 3: adjust Bi-plane Classifier according to updated Positive/Negative Sets
        self.bi_plane.adjust_classifier_plane(frame_ID)






