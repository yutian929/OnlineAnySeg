import numpy as np
import torch
from sklearn.linear_model import SGDClassifier


class Mask_pair:
    def __init__(self, mask_index, pair_iou, pair_geo_sim, pair_sem_sim, pair_score=0., add_frame_ID=0):
        self.mask_index = mask_index

        self.pair_iou = pair_iou
        self.pair_geo_sim = pair_geo_sim
        self.pair_sem_sim = pair_sem_sim

        self.pair_score = pair_score
        self.pair_score_new = 0.
        self.add_frame_ID = add_frame_ID

    def fill_new_score(self, new_score):
        self.pair_score_new = new_score

    @property
    def get_raw_scores(self):
        return np.array([self.pair_iou, self.pair_geo_sim, self.pair_sem_sim])


class Bi_Plane_Classifier:
    def __init__(self, cfg, device="cuda:0"):
        self.cfg = cfg
        self.device = device

        self.trust_area = []
        self.hesitate_area = []
        self.max_merged_max_num = self.cfg["scene"]["mask_num"] // 2
        self.hes_mask_IDs = -1 * torch.ones((self.max_merged_max_num, 2), dtype=torch.int64)  # Hesitate Area中mask_index到真正的 merged_mask_ID 的映射表 (避免每次mask merging后要更新2个area中的 Mask_pair objs)

        self.ta_thresh = self.cfg["classifier"]["trust_thresh"]  # threshold for belonging to Trust Area (may need to adjust)
        self.hesitate_thresh = self.cfg["classifier"]["hesitate_thresh"]  # threshold for belonging to Hesitate Area (may need to adjust)

        # record IoU/semantic similarity/geometric similarity in this iteration of mask merging (updated in each merging)
        self.candi_iou = None
        self.candi_sem_sim = None
        self.candi_geo_sim = None

        self.positive_set = []
        self.negative_set = []
        self.positive_set_iter = []
        self.negative_set_iter = []

        self.clf = self.initialize_classifier()  # *** classifier plane
        self.positive_set_acc = []
        self.negative_set_acc = []
        self.update_clf_flag = False

    @property
    def get_hes_mask_pair_IDs(self):
        hes_mask_num = len(self.hesitate_area)
        return self.hes_mask_IDs[:hes_mask_num]

    @property
    def get_plane_parameters(self):
        w = self.clf.coef_[0]
        b = self.clf.intercept_[0]
        A, B, C, d = w[0].item(), w[1].item(), w[2].item(), b.item()
        return (A, B, C, d)

    def initialize_classifier(self):
        A, B, C, d = self.cfg["classifier"]["A"], self.cfg["classifier"]["B"], self.cfg["classifier"]["C"], self.cfg["classifier"]["d"]  # bi-plane parameters (format: Ax+By+Cz+d=0)

        # **构造初始超平面（法向量 w 和 偏置 b）**
        w_init = np.array([A, B, C], dtype=np.float64)
        b_init = np.array([d], dtype=np.float64)

        # **初始化分类器（不对称权重SVM + 增量学习）**
        pos_weight = self.cfg["classifier"]["pos_w"]  # weight for positive samples in classifier update
        neg_weight = self.cfg["classifier"]["neg_w"]  # weight for positive samples in classifier update
        clf = SGDClassifier(loss="hinge", max_iter=self.cfg["classifier"]["max_iter"], tol=1e-3, class_weight={1: pos_weight, -1: neg_weight})

        # **手动设置分类器的初始参数**
        clf.coef_ = np.array([w_init], dtype=np.float64)
        clf.intercept_ = np.array(b_init, dtype=np.float64)
        clf.classes_ = np.array([-1, 1])  # labels（negative samples: -1, positive samples: 1）
        return clf

    # @brief:
    # @param
    # @param
    # @param
    #-@return
    def do_classification(self, x_mat, y_mat, z_mat):
        A, B, C, d = self.get_plane_parameters  # parameters of Ax+By+Cy+z=0
        positive_mask = (A * x_mat + B * y_mat + C * z_mat + d > 0)
        return positive_mask

    def judge_trust_area(self, candi_iou, candi_sem_sim, candi_geo_sim):
        ta_mask = (candi_iou + candi_sem_sim + candi_geo_sim) >= self.ta_thresh  # Trust Area的判断方程
        ta_indices = torch.where(ta_mask)[0]
        return ta_mask, ta_indices

    def judge_hesitate_area(self, candi_iou, candi_sem_sim, candi_geo_sim):
        h_mask1 = (candi_iou + candi_sem_sim + candi_geo_sim) >= self.hesitate_thresh
        h_mask2 = (candi_iou + candi_sem_sim + candi_geo_sim) < self.ta_thresh
        h_mask = h_mask1 & h_mask2
        h_indices = torch.where(h_mask)[0]
        return h_mask, h_indices

    def judge_discard_area(self, candi_iou, candi_sem_sim, candi_geo_sim):
        da_mask = (candi_iou + candi_sem_sim + candi_geo_sim) < self.hesitate_thresh  # Trust Area的判断方程
        da_indices = torch.where(da_mask)[0]
        return da_mask, da_indices


    # @brief: giving N mask pairs, for each pair judge whether it belongs to Trust Area or Hesitate Area;
    # @param iou_mat: IoU matrix of all current mask, Tensor(mask_num, mask_num);
    # @param sem_feature_sim_mat: Semantic Similarity Matrix of all current mask, Tensor(mask_num, mask_num);
    # @param geo_feature_sim_mat: Geometric Similarity Matrix of all current mask, Tensor(mask_num, mask_num);
    # @param candi_row_indices: mask_ID_1 list of candidate mask pairs, Tensor(pair_num, );
    # @param candi_col_indices: mask_ID_2 list of candidate mask paris, Tensor(pair_num, ).
    def judge_pair_area(self, iou_mat, sem_feature_sim_mat, geo_feature_sim_mat, candi_row_indices, candi_col_indices):
        # Step 1: extract each pair's respective score
        candi_iou = iou_mat[candi_row_indices, candi_col_indices]
        candi_sem_sim = sem_feature_sim_mat[candi_row_indices, candi_col_indices]
        candi_geo_sim = geo_feature_sim_mat[candi_row_indices, candi_col_indices]
        candi_sim = candi_iou + candi_sem_sim + candi_geo_sim  # total scores for all candidate mask pairs

        self.candi_iou, self.candi_sem_sim, self.candi_geo_sim = candi_iou, candi_sem_sim, candi_geo_sim

        # Step 2: judge belonging area for each candidate mask pair
        t_mask, t_indices = self.judge_trust_area(candi_iou, candi_sem_sim, candi_geo_sim)  # judge whether each pair belongs to Trust Area
        h_mask, h_indices = self.judge_hesitate_area(candi_iou, candi_sem_sim, candi_geo_sim)
        d_mask, d_indices = self.judge_discard_area(candi_iou, candi_sem_sim, candi_geo_sim)
        return t_mask, h_mask, d_mask, candi_sim


    # brief: for all currently existing masks, compute the 3 scores for each candidate mask pair;
    # @param iou_mat: IoU of all currently existing mask pairs, Tensor(c_mask_num, c_mask_num), dtype=float32;
    # @param sem_feature_sim_mat: Semantic features of all currently existing mask pairs, Tensor(c_mask_num, c_mask_num), dtype=float32;
    # @param geo_feature_sim_mat: Geometric features of all currently existing mask pairs, Tensor(c_mask_num, c_mask_num), dtype=float32;
    # @param candi_pair_mask: Mask of all candidate mask pairs, Tensor(c_mask_num, c_mask_num), dtype=bool.
    def judge_candidate_pairs(self, frame_ID, iou_mat, sem_feature_sim_mat, geo_feature_sim_mat, candi_pair_mask):
        # Step 1: extract mask_IDs for each candidate mask pair: (mask_ID_1, mask_ID_2)
        candi_row_indices, candi_col_indices = torch.where(candi_pair_mask)  # row_ID/col_ID of each candidate mask pair, Tensor(pair_num, ), Tensor(pair_num, )
        candi_pair_indices = torch.stack([candi_row_indices, candi_col_indices], dim=-1)  # each candaite pair's (mask_ID_1, mask_ID_2), Tensor(pair_num, 2)

        # Step 2: for each candidate mask pair, judge which area (Trust/Hesitate/Discard Area) it belongs to
        t_mask, h_mask, d_mask, candi_sim = self.judge_pair_area(iou_mat, sem_feature_sim_mat, geo_feature_sim_mat, candi_row_indices, candi_col_indices)
        t_pair_indices = candi_pair_indices[t_mask]  # mask pairs that belong to Trust Area, Tensor(T_pari_num, 2)
        h_pair_indices = candi_pair_indices[h_mask]  # mask pairs that belong to Hesitating Area, Tensor(T_pair_num, 2)
        d_pair_indices = candi_pair_indices[d_mask]
        t_pair_scores, h_pair_scores, d_pair_scores = candi_sim[t_mask], candi_sim[h_mask], candi_sim[d_mask]

        # Step 3: 记录犹豫区中的所有mask pairs (对应的merged_mask_IDs); 在之后本轮merge完后要对这些犹豫区的mask pairs再次计算他们新的得分
        if len(self.hesitate_area) > 0:
            self.hesitate_area.clear()

        h_pair_ious = self.candi_iou[h_mask]
        h_pair_geo_sims = self.candi_geo_sim[h_mask]
        h_pair_sem_sims = self.candi_sem_sim[h_mask]
        for i, (h_pair, h_pair_score, h_pair_iou, h_pair_geo_sim, h_pair_sem_sim) in enumerate( zip(h_pair_indices, h_pair_scores, h_pair_ious, h_pair_geo_sims, h_pair_sem_sims) ):
            self.hesitate_area.append( Mask_pair(i, h_pair_iou.item(), h_pair_geo_sim.item(), h_pair_sem_sim.item(), h_pair_score.item(), frame_ID) )
            self.hes_mask_IDs[i] = h_pair


    # @brief: for each mask pair in Hesitate Area, re-judge which area it belongs to after this iteration of merging
    def rejudge_hesitate_pairs(self, frame_ID, iou_mat, sem_feature_sim_mat, geo_feature_sim_mat):
        # Step 1: for all mask pairs in Hesitate Area currently, recompute its score and rejudge which area it belongs to after this iteration of merging
        hes_mask_pair_IDs = self.get_hes_mask_pair_IDs  # Tensor(hes_pair_num, 2)
        hes_pair_row_indices, hes_pair_col_indices = hes_mask_pair_IDs[:, 0], hes_mask_pair_IDs[:, 1]
        t_mask, h_mask, d_mask, hes_pair_scores = self.judge_pair_area(iou_mat, sem_feature_sim_mat, geo_feature_sim_mat, hes_pair_row_indices, hes_pair_col_indices)

        area_mask = torch.stack([t_mask, h_mask, d_mask], dim=-1).float()  # Tensor(hes_pair_num, 3), dtype=float32
        area_indices = torch.argmax(area_mask, dim=-1)  # indicate which area(Trust/Hesidate/Discard) each candidate mask belongs to, Tensor(hes_pair_num, )

        # Step 2: for each mask, judge whether it should be appended into Positive Set or Negative Set
        self.positive_set_iter.clear()
        self.negative_set_iter.clear()
        for (hes_mask_pair_ID, hes_pair, hes_pair_score, area_index) in zip(hes_mask_pair_IDs, self.hesitate_area, hes_pair_scores, area_indices):
            hes_pair.fill_new_score(hes_pair_score.item())
            if area_index == 0:
                self.positive_set_iter.append(hes_pair)
            elif area_index == 2:
                self.negative_set_iter.append(hes_pair)

        self.positive_set.extend(self.positive_set_iter)
        self.negative_set.extend(self.negative_set_iter)

        self.positive_set_acc.extend(self.positive_set_iter)
        self.negative_set_acc.extend(self.negative_set_iter)
        if len(self.negative_set_iter) > 0:
            self.update_clf_flag = True

        print( "(TEST) After merging of frame_%d, %d pairs are added to Positive Set, %d pairs are added to Negative Set" % ( frame_ID, len(self.positive_set_iter), len(self.negative_set_iter) ) )


    # @brief: adjust the parameters of classifier plane according to updated Positive/Negative Sets
    def adjust_classifier_plane(self, frame_ID):
        if not self.update_clf_flag:
            return

        # Step 1: extract coords of positive and negative samples
        positive_coords = [mask_pair.get_raw_scores for mask_pair in self.positive_set_acc]
        negative_coords = [mask_pair.get_raw_scores for mask_pair in self.negative_set_acc]

        X_new = np.vstack(positive_coords+negative_coords)  # ndarray(n, 3)
        y_new = np.hstack((np.ones(len(positive_coords)), -np.ones(len(negative_coords))))

        # Step 2: perform incremental update of the classifier
        plane_bf = self.plane_formulation(self.get_plane_parameters)
        print("Frame_%d: (Before) plane: %s" % (frame_ID, plane_bf))

        self.clf.partial_fit(X_new, y_new, classes=[-1, 1])  # 仅基于新数据调整分类平面

        plane_aft = self.plane_formulation(self.get_plane_parameters)
        print("Frame_%d: (After) plane: %s" % (frame_ID, plane_aft))

        # Step 3: reset the corresponding vars
        self.update_clf_flag = False
        self.positive_set_acc.clear()
        self.negative_set_acc.clear()


    ################################ For testing ################################

    def plane_formulation(self, parameters):
        A, B, C, d = parameters[0], parameters[1], parameters[2], parameters[3]
        plane_formulation = "(%.2f) * x + (%.2f) * y + (%.2f) * z + (%.2f) = 0" % (A, B, C, d)
        return plane_formulation
