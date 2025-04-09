import os
import copy
from typing import Union, Dict
import numpy as np
import torch

# from graph_rep import graph_rep
# from GS_scene import GSScene_local
from tool.geometric_helpers import pts_transform, pts_transform_multi_poses, compute_projected_pts, compute_projected_pts_multi,\
    compute_visibility_mask_ids, compute_visibility_masks_ids, compute_complementary, save_pts_ply, save_projection_image, \
    save_seg_image, save_seg_image_select_mask


# @brief: 计算各local instances和当前已重建场景中global instances的重叠关系;
# @param masked_pts: 每个local instance在Global PC中的point mask, Tensor(l_instance_num, g_s_pts_num), dtype=int32;
# @param instance_pt_count: 已重建场景中每个global instance的point mask, Tensor(g_instance_num, g_s_pts_num), dtype=int32;
#-@return iou_matrix: 当前帧中每个valid instance和场景中已重建的每个instance的point数IoU, Tensor(v_instance_num, e_instance_num), dtype=float32;
#-@return precision_matrix: 每个intersection中点的数量与对应已存在的instance中点数的比值, Tensor(v_instance_num, e_instance_num), dtype=float32;
#-@return recall_matrix: 每个intersection中点的数量与对应当前帧中valid instance中点数的比值, Tensor(v_instance_num, e_instance_num), dtype=float32.
def get_relation_matrix(masked_pts, instance_pt_count):
    instance_pt_mask = instance_pt_count.to(torch.float32)  # 已重建部分点云中在当前帧下的visible mask(0/1), Tensor(e_instance_num, pts_num);
    masked_pts = masked_pts.to(torch.float32)  # 元素为0/1

    intersection = masked_pts @ instance_pt_mask.T  # 当前帧中每个valid instance和场景中已重建的每个instance重合的point数, Tensor(v_instance_num, e_instance_num)
    masked_pts_sum = masked_pts.sum(1, keepdims=True)  # 当前帧中各valid instances所包含的点数, Tensor(v_instance_num, 1)
    instance_pt_mask_sum = instance_pt_mask.sum(1, keepdims=True)  # 当前场景中已存在的各instances所包含的点数, Tensor(e_instance_num, 1)

    union = masked_pts_sum + instance_pt_mask_sum.T - intersection  # 当前帧中每个valid instance和场景中已存在的每个instance的union point数, Tensor(v_instance_num, e_instance_num)
    iou_matrix = intersection / (union + 1e-6)  # 当前帧中每个valid instance和场景中已重建的每个instance的point数IoU, Tensor(v_instance_num, e_instance_num)
    precision_matrix = intersection / (instance_pt_mask_sum.T + 1e-6)  # 每个intersection中点的数量与对应已存在的instance中点数的比值, Tensor(v_instance_num, e_instance_num)
    recall_matrix = intersection / (masked_pts_sum + 1e-6)  # 每个intersection中点的数量与对应当前帧中valid instance中点数的比值, Tensor(v_instance_num, e_instance_num)
    return intersection, iou_matrix, precision_matrix, recall_matrix


def find_correspondences(iou_matrix, recall_matrix, parti_local_instance_ids, parti_global_instance_ids, iou_threshold=0.3, recall_threshould=0.7, unmatched_indicator=-1):
    local_instance_num = iou_matrix.shape[0]
    correspondences = torch.full((local_instance_num, ), fill_value=-1)

    out = torch.sort(recall_matrix, dim=unmatched_indicator, descending=True)
    corre_max_recall = out[0][:, 0]
    corre_max_global_instance_indices = out[1][:, 0].to(correspondences)

    corre_max_global_instance_ids = parti_global_instance_ids[corre_max_global_instance_indices]
    correspondences = torch.where(corre_max_recall > recall_threshould, corre_max_global_instance_ids, correspondences)
    return correspondences


# @brief: 把一组给定的3D点投影到个给定的view上;
# @param pts_xyz: points needed to be transformed, Tensor(pts_num, 3);
# @param pose2_c2w: pose (c2w) to apply transformations respectively, Tensor(4, 4);
# @param intrinsic: Tensor(3, 3);
# @param depth_images: Tensor(H, W);
# @param seg_images: Tensor(H, W);
#-@return pixels_uv: Tensor(pts_num, 2), dtype=int64;
#-@return visibility_masks: Tensor(pts_num, ), dtype=bool;
#-@return projected_pts_mask_id: Tensor(pts_num, ), dtype=int64.
def project_to_view(pts_xyz, pose_c2w, intrinsic, depth_image, seg_image):
    # Step 1: transform all points to each of given image space
    pose_w2c = torch.inverse(pose_c2w)
    pts_xyz_cam = pts_transform(pts_xyz, pose_w2c)  # Tensor(N, 3)
    pixels_uv, front_mask = compute_projected_pts(pts_xyz_cam, intrinsic)  # Tensor(pts_num, 2) / Tensor(pts_num, ), dtype=bool

    # Step 2: judge whether each projected point has valid uv and valid depth
    visibility_mask, projected_pts_mask_id = compute_visibility_mask_ids(pts_xyz_cam, pixels_uv, depth_image, seg_image, front_mask)
    return pixels_uv, visibility_mask, projected_pts_mask_id


# @brief: 把一组给定的3D点投影到n个给定的views上;
# @param pts_xyz: points needed to be transformed, Tensor(pts_num, 3);
# @param poses_c2w: all poses (c2w) to apply transformations respectively, Tensor(K, 4, 4);
# @param intrinsic: Tensor(3, 3);
# @param depth_images: Tensor(K, H, W);
# @param seg_images: Tensor(K, H, W);
#-@return pixels_uv: Tensor(K, pts_num, 2), dtype=int64;
#-@return visibility_masks: Tensor(K, pts_num), dtype=bool;
#-@return projected_pts_mask_id: Tensor(K, pts_num), dtype=int64.
def project_to_views(pts_xyz, poses_c2w, intrinsic, depth_images, seg_images):
    # Step 1: transform all points to each of given image space
    poses_w2c = torch.inverse(poses_c2w)
    pts_xyz_cam = pts_transform_multi_poses(pts_xyz, poses_w2c)  # Tensor(K, N, 3)
    pixels_uv, front_masks = compute_projected_pts_multi(pts_xyz_cam, intrinsic)  # Tensor(K, pts_num, 2) / Tensor(K, pts_num)

    # Step 2: judge whether each projected point has valid uv and valid depth
    visibility_masks, projected_pts_mask_id = compute_visibility_masks_ids(pts_xyz_cam, pixels_uv, depth_images, seg_images, front_masks)  # Tensor(K, pts_num)
    return pixels_uv, visibility_masks, projected_pts_mask_id


# @brief: project a given pointcloud to K different views. 在每个view中，我们记录 投影到GT mask上的点 / 所有点 的比值(hit ratio);
# @param pts_xyz: points needed to be transformed, Tensor(N, 3), device=cuda;
# @param poses_c2w: all poses (c2w) to apply transformations respectively, Tensor(K, 4, 4), device=cuda;
# @param intrinsic: Tensor(3, 3), device=cuda;
# @param depth_images: Tensor(K, H, W), dtype=float32, device=cuda;
# @param seg_images: Tensor(K, H, W), dtype=uint8, device=cuda;
# @param view_mask_ids: Tensor(K, ), dtype=uint8, device=cuda;
#-@return: Tensor(K, ), dtype=float32;
def compute_project_scores(pts_xyz, poses_c2w, intrinsic, depth_images, seg_images, view_mask_ids):
    pixels_uv, visibility_masks, projected_pts_mask_id = project_to_views(pts_xyz, poses_c2w, intrinsic, depth_images, seg_images)
    visible_pts_num = torch.sum(visibility_masks, dim=-1)
    projected_pts_mask_id = projected_pts_mask_id * visibility_masks  # Tensor(K, pts_num), dtype=uint8, device=cuda

    views_hit_pts_num = torch.sum( projected_pts_mask_id == view_mask_ids.unsqueeze(-1), dim=-1 )
    views_hit_ratio = torch.where(visible_pts_num > 0, views_hit_pts_num / visible_pts_num, torch.zeros_like(views_hit_pts_num).float())
    return views_hit_ratio


# @brief: project a given pointcloud to K different views. 在每个view中，我们记录 (1)该instance在该view下的visibility ratio; (2)投影到GT mask上的点 / 所有点 的比值(hit ratio);
# @param pts_xyz: points needed to be transformed, Tensor(N, 3), device=cuda;
# @param poses_c2w: all poses (c2w) to apply transformations respectively, Tensor(K, 4, 4), device=cuda;
# @param intrinsic: Tensor(3, 3), device=cuda;
# @param depth_images: Tensor(K, H, W), dtype=float32, device=cuda;
# @param seg_images: Tensor(K, H, W), dtype=uint8, device=cuda;
# @param view_mask_ids: target mask_id in each given view, Tensor(K, ), dtype=uint8, device=cuda;
#-@return: Tensor(K, ), dtype=float32;
def compute_iv_project_scores(pts_xyz, poses_c2w, intrinsic, depth_images, seg_images, view_mask_ids, k_vis=0.5, save_image=False, save_dir=None):
    pixels_uv, visibility_masks, projected_pts_mask_id = project_to_views(pts_xyz, poses_c2w, intrinsic, depth_images, seg_images)
    visible_pts_num = torch.sum(visibility_masks, dim=-1)
    projected_pts_mask_id = projected_pts_mask_id * visibility_masks  # Tensor(K, pts_num), dtype=uint8, device=cuda

    # (1) visibility ratio
    instance_size = pts_xyz.shape[0]
    views_vis_ratio = visible_pts_num / instance_size

    # (2) avg hit ratio
    # 2.1: recall
    views_hit_pts_num = torch.sum( projected_pts_mask_id == view_mask_ids.unsqueeze(-1), dim=-1 )
    views_hit_ratio = torch.where(visible_pts_num > 0, views_hit_pts_num / visible_pts_num, torch.zeros_like(views_hit_pts_num).float())

    # # 2.2: precision
    # seg_imgs_flatten = torch.flatten(seg_images, start_dim=1)
    # view_mask_count_mask = torch.where(seg_imgs_flatten == view_mask_ids.unsqueeze(-1), torch.ones_like(seg_imgs_flatten), torch.zeros_like(seg_imgs_flatten))
    # view_mask_size = view_mask_count_mask.sum(-1)  # pixel size of each target mask, Tensor(K, )
    # views_precision = torch.where(view_mask_size > 0, views_hit_pts_num / view_mask_size, torch.zeros_like(views_hit_pts_num).float())
    # views_hit_ratio = (views_recall + views_precision) / 2

    final_scores = k_vis * views_vis_ratio + (1 - k_vis) * views_hit_ratio
    return final_scores


# @brief: project a given pointcloud to K different views. 在每个view中，我们记录该instance投影到给定帧上重合率最大的mask, 以及在该mask上的投影得分;
# @param pts_xyz: points needed to be transformed, Tensor(N, 3), device=cuda;
# @param poses_c2w: all poses (c2w) to apply transformations respectively, Tensor(K, 4, 4), device=cuda;
# @param intrinsic: Tensor(3, 3), device=cuda;
# @param depth_images: Tensor(K, H, W), dtype=float32, device=cuda;
# @param seg_images: Tensor(K, H, W), dtype=uint8, device=cuda;
# @param view_mask_ids: target mask_id in each given view, Tensor(K, ), dtype=uint8, device=cuda;
#-@return: Tensor(K, ), dtype=float32;
def compute_iv_project_scores_mask(pts_xyz, poses_c2w, intrinsic, depth_images, seg_images, mask_visible_threshold=0.3, contained_threshold=0.8, device="cuda:0"):
    # Step 1: project to each given view
    pixels_uv, visibility_masks, projected_pts_mask_id = project_to_views(pts_xyz, poses_c2w, intrinsic, depth_images, seg_images)
    projected_pts_mask_id = projected_pts_mask_id * visibility_masks  # 该3D点集投影到各帧上各自投影对应的mask分布情况, Tensor(K, pts_num), dtype=uint8, device=cuda
    views_visible_pts_num = torch.sum(projected_pts_mask_id.bool() & visibility_masks, dim=-1)

    # Step 2: for each view, compute its corresponding projected mask, and projecting score
    view_num = poses_c2w.shape[0]
    pts_num = pts_xyz.shape[0]
    view_mask_ids = torch.zeros((view_num, )).to(device)
    view_visible_ratios = torch.zeros((view_num, )).to(device)
    view_recalls = torch.zeros((view_num, )).to(device)

    for i in range(view_num):
        visible_pts_num = views_visible_pts_num[i]
        visible_ratio = visible_pts_num / pts_num
        if visible_ratio < mask_visible_threshold:
            continue

        mask_id_count = torch.bincount(projected_pts_mask_id[i])
        mask_id_count[0] = 0  # 不考虑该mask中那些落在当前帧背景(mask_id==0)上的3D点
        max_mask_id = torch.argmax(mask_id_count)  # 该mask在当前帧上绝大部分点落在的那个mask_id
        # max_mask_size = torch.count_nonzero( seg_images[i] == max_mask_id.to(seg_images) )
        contained_ratio = mask_id_count[max_mask_id] / torch.sum(mask_id_count)  # 绝大部分点落在某一个mask_id上，这个"绝大部分"具体是百分之多少
        if contained_ratio > contained_threshold:
            view_mask_ids[i] = max_mask_id
            view_visible_ratios[i] = visible_ratio
            view_recalls[i] = mask_id_count[max_mask_id] / visible_pts_num
    # END for

    return view_mask_ids, view_visible_ratios, view_recalls


def find_instance_correspondences(score_mat):
    mask_num = score_mat.shape[0]
    correspondences = torch.full((mask_num, ), fill_value=-1)

    out = torch.sort(score_mat, dim=-1, descending=True)
    corre_max_scores = out[0][:, 0]
    corre_max_instance_ids = out[1][:, 0].to(correspondences)

    correspondences = torch.where(corre_max_scores > 0, corre_max_instance_ids, correspondences)
    return correspondences


def find_mask_local_correspondences(score_mat, corre_mat):
    mask_num = score_mat.shape[0]
    correspondences = torch.full((mask_num, ), fill_value=-1)

    out = torch.sort(score_mat * corre_mat, dim=-1, descending=True)
    corre_max_scores = out[0][:, 0]
    corre_max_instance_ids = out[1][:, 0].to(correspondences)

    correspondences = torch.where(corre_max_scores > 0, corre_max_instance_ids, correspondences)
    return correspondences


# @brief:
def mask_instance_matching(mask_graph, gs_scene_local, intersect_threshold, precision_threshold, recall_threshold, k=0.5, top_k_views=5):
    # Step 1: preparation
    local_pts = gs_scene_local.get_xyz  # Tensor(l_pts_num, 3)
    local_pts_num = gs_scene_local.get_pts_num
    l_instance_pts = gs_scene_local.get_instance_point_mask  # Tensor(l_instance_num, l_pts_num)

    mg_instance_num = len(mask_graph.mask_points)
    mg_instance_pts = torch.zeros((mg_instance_num, local_pts_num)).to(l_instance_pts)  # Tensor(mg_instance_num, l_pts_num)
    for i in range(mg_instance_num):
        mg_instance_pts[i, mask_graph.mask_points[i]] = 1

    # Step 2: find correspondences
    intersection, _, precision_matrix, recall_matrix = get_relation_matrix(mg_instance_pts, l_instance_pts)
    condition_mat = (precision_matrix > precision_threshold) | (recall_matrix > recall_threshold) | (intersection > intersect_threshold)
    corre_mat_c = torch.where(condition_mat, torch.ones_like(recall_matrix), torch.zeros_like(recall_matrix))  # 0/1, Tensor(mg_instance_num, l_instance_num)
    score_mat = (k * recall_matrix + (1-k) * precision_matrix) * corre_mat_c

    p_mask_indices, p_instance_indices = torch.where(corre_mat_c > 0)
    mask_correspondences = find_mask_local_correspondences(score_mat, corre_mat_c)

    return mg_instance_pts, mask_correspondences


# @brief: for each given 3D instance, compute top K view according to precision and recall;
# @param instance_view_precision: 各3D instance在各帧上 落在对应的mask中的点数 / 对应mask本身的点数, Tensor(instance_num, frame_num);
# @param instance_view_recall: 各3D instance在各帧上 落在对应的mask中的点数 / 对应mask本身的点数, Tensor(instance_num, frame_num);
#-@return instance_view_scores: 各个instance在各个view下的score(已降序排列好), Tensor(instance_num, frame_num);
def compute_instance_view_project_scores(instance_view_precision, instance_view_recall, k=0.5):
    instance_view_scores = k * instance_view_precision + (1 - k) * instance_view_recall
    return instance_view_scores


# @brief: for each given 3D instance, compute top K view according to precision and recall;
# @param instance_view_precision: 各3D instance在各帧上 落在对应的mask中的点数 / 对应mask本身的点数, Tensor(instance_num, frame_num);
# @param instance_view_recall: 各3D instance在各帧上 落在对应的mask中的点数 / 对应mask本身的点数, Tensor(instance_num, frame_num);
#-@return instance_view_scores: 各个instance在各个view下的score(已降序排列好), Tensor(instance_num, frame_num);
def compute_instance_view_visible_scores(instance_view_visible_ratio, instance_view_precision, instance_view_recall, k=0.5):
    instance_view_scores = k * (instance_view_precision + instance_view_recall) / 2 + (1 - k) * instance_view_visible_ratio
    return instance_view_scores


# @brief: 对于Global PC中的某些instance和和Local PC中的某些instance, 通过cross-project的方式，注意判断每个pair是否是对应关系;
# @param gs_scene:
# @param gs_scene_local:
# @param global_instance_ids:
# @param local_instance_ids:
# @param im_database:
# @param intrinsic: Tensor(3, 3)；
#-@return l_g_instance_score: 每个local instance和global instance pair之间的交叉投影得分;
#-@return l_g_instance_corr_mat:
def cross_project_lg(gs_scene, gs_scene_local, global_instance_ids, local_instance_ids, im_database, intrinsic,
                     top_k_g=3, top_k_l=5, score_threshold=0.7, cp_score_threshold=0.7):
    # Step 1: extract points coordinates of each selected global instance and local instance
    local_instances_xyz = [ gs_scene_local.get_xyz[gs_scene_local.get_instance_point_mask[local_instance_id].bool()] for local_instance_id in local_instance_ids]
    global_instances_xyz = [ gs_scene.get_xyz[gs_scene.get_instance_point_mask[global_instance_id].bool()] for global_instance_id in global_instance_ids ]

    # Step 2: for each local instance, compute its cross-projection score with each of global instance
    l_g_instance_score = torch.zeros((local_instance_ids.shape[0], global_instance_ids.shape[0]))
    for i in range(local_instance_ids.shape[0]):
        # 2.1: compute top K views for this local instance
        local_instance_xyz = local_instances_xyz[i]
        l_instance_id_this = local_instance_ids[i]  # 该local instance的id (不是在形参local_instance_ids中的indices)
        l_instance_view_scores = compute_instance_view_project_scores(gs_scene_local.instance_frame_precision[l_instance_id_this], gs_scene_local.instance_frame_recall[l_instance_id_this])
        if l_instance_view_scores.max() < score_threshold:
            continue
        k_local = torch.count_nonzero(l_instance_view_scores >= score_threshold)  # 对于该local instance, 取其前多少个views
        k_local = min(k_local.item(), top_k_l)
        l_instance_top_k_views = gs_scene_local.instance_view_indices[l_instance_id_this][:k_local]
        l_instance_top_k_view_mask_ids = gs_scene_local.instance_frame_mask_ids[l_instance_id_this][:k_local]
        l_instance_top_k_depth = gs_scene_local.depth_imgs[l_instance_top_k_views]
        l_instance_top_k_seg = gs_scene_local.seg_imgs[l_instance_top_k_views]
        l_instance_top_k_poses = gs_scene_local.poses_c2w[l_instance_top_k_views]

        for j in range(global_instance_ids.shape[0]):
            # 2.1: compute top K views for this global instance
            global_instance_xyz = global_instances_xyz[j]
            g_instance_id_this = global_instance_ids[j]  # 该global instance的id (不是在形参global_instance_ids中的indices)

            g_instance_view_scores = im_database.instance_mask_scores[g_instance_id_this]
            if g_instance_view_scores[0] < score_threshold:
                continue
            k = torch.count_nonzero(g_instance_view_scores >= score_threshold)  # 对于该global instance, 取其前多少个views
            k = min(k.item(), top_k_g)
            g_instance_top_k_views = im_database.instance_mask_views[g_instance_id_this][:k]
            g_instance_top_k_view_mask_ids = im_database.instance_mask_ids[g_instance_id_this][:k]
            g_instance_top_k_depth = im_database.kfSet.kf_depth_imgs[g_instance_top_k_views]
            g_instance_top_k_seg = im_database.kfSet.kf_seg_imgs[g_instance_top_k_views]
            g_instance_top_k_poses = im_database.kfSet.poses_c2w[g_instance_top_k_views]

            # 2.2: project this local instance to global instance's top K views
            view_scores_l2g = compute_project_scores(local_instance_xyz, g_instance_top_k_poses, intrinsic, g_instance_top_k_depth, g_instance_top_k_seg, g_instance_top_k_view_mask_ids)
            avg_scores_l2g = torch.mean(view_scores_l2g)

            # 2.3: project this global instance to local instance's top K views
            view_scores_g2l = compute_project_scores(global_instance_xyz, l_instance_top_k_poses, intrinsic, l_instance_top_k_depth, l_instance_top_k_seg, l_instance_top_k_view_mask_ids)
            avg_scores_g2l = torch.mean(view_scores_g2l)

            if avg_scores_l2g >= cp_score_threshold and avg_scores_g2l >= cp_score_threshold:
                l_g_instance_score[i][j] = (avg_scores_l2g + avg_scores_g2l) / 2

    l_g_instance_corr_mat = l_g_instance_score.bool().float()
    return l_g_instance_score, l_g_instance_corr_mat


# @brief: for some of global instances, check whether each pair of them belongs to 1 instance by cross-projection;
# @param check_condition_mat: Tensor(g_instance_num, g_instance_num), dtype=bool();
# @param global_instance_xyz_list: list of Tensor(n_i, 3);
# @param involved_g_instance_ids: participated global instances, Tensor(i_g_instance_num, );
# @param intrinsics: Tensor(3, 3);
# @param im_database:
#-@return: correspondences matrix of all global instances, Tensor(g_instance_num, g_instance_num). dtypr=bool.
def check_instance_corr_by_cross_project(check_condition_mat, global_instance_xyz_list, involved_g_instance_ids, intrinsics, im_database, \
                                         top_k=5, score_threshold=0.7, cp_score_threshold=0.7):
    # Step 1: compute potential global instance pairs that need to check
    involved_instance_num = involved_g_instance_ids.shape[0]  # ids of global instances contained by current local PC

    # Step 2: for each potential pair, do cross-projecting to compute scores
    # 2.1: for each involved global instance, extract its corresponding vars
    g_instance_top_k_kf_ids_list = []
    g_instance_top_k_view_mask_ids_list = []
    g_instance_top_k_depth_list = []
    g_instance_top_k_seg_list = []
    g_instance_top_k_poses_list = []
    g_instance_k_num_list = []

    for i in range(involved_instance_num):
        g_instance_id_this = involved_g_instance_ids[i]
        g_instance_view_scores = im_database.instance_mask_scores[g_instance_id_this]
        if g_instance_view_scores[0] < score_threshold:
            g_instance_top_k_kf_ids_list.append(None)
            g_instance_top_k_view_mask_ids_list.append(None)
            g_instance_top_k_depth_list.append(None)
            g_instance_top_k_seg_list.append(None)
            g_instance_top_k_poses_list.append(None)
            g_instance_k_num_list.append(0)
        else:
            k = torch.count_nonzero(g_instance_view_scores >= score_threshold)  # 对于该global instance, 取其前多少个views
            k = min(k.item(), top_k)
            g_instance_top_k_views = im_database.instance_mask_views[g_instance_id_this][:k]  # 该instance所关联的top K个views的keyframe_ids
            g_instance_top_k_kf_ids_list.append(g_instance_top_k_views)
            g_instance_top_k_view_mask_ids_list.append( im_database.instance_mask_ids[g_instance_id_this][:k] )
            g_instance_top_k_depth_list.append( im_database.kfSet.kf_depth_imgs[g_instance_top_k_views] )
            g_instance_top_k_seg_list.append( im_database.kfSet.kf_seg_imgs[g_instance_top_k_views] )
            g_instance_top_k_poses_list.append( im_database.kfSet.poses_c2w[g_instance_top_k_views] )
            g_instance_k_num_list.append(k)

    # 2.2 compute each pair's cross-projecting score
    g_instance_scores = torch.zeros_like(check_condition_mat).float()  # Tensor(inv_g_instance_num, inv_g_instance_num)
    for i in range(involved_instance_num):
        for j in range(involved_instance_num):
            if j <= i:
                continue
            # 2.2.1: check whether both global instances in this pair are valid
            g_instance_source_id = involved_g_instance_ids[i]
            g_instance_target_id = involved_g_instance_ids[j]
            if not check_condition_mat[g_instance_source_id, g_instance_target_id] and not check_condition_mat[g_instance_target_id, g_instance_source_id]:
                continue
            if g_instance_k_num_list[i] == 0 or g_instance_k_num_list[j] == 0:
                continue

            # 2.2.2: extract corresponding var of both global instances
            source_g_instance_xyz = global_instance_xyz_list[i]
            target_g_instance_xyz = global_instance_xyz_list[j]

            # 2.2.3: project source global instance to target global instance's top K views
            view_scores_s2t = compute_project_scores(source_g_instance_xyz, g_instance_top_k_poses_list[j], intrinsics, g_instance_top_k_depth_list[j],
                                                     g_instance_top_k_seg_list[j], g_instance_top_k_view_mask_ids_list[j])
            avg_scores_s2t = torch.mean(view_scores_s2t)

            # 2.2.4: project target global instance to source global instance's top K views
            view_scores_t2s = compute_project_scores(target_g_instance_xyz, g_instance_top_k_poses_list[i], intrinsics, g_instance_top_k_depth_list[i],
                                                     g_instance_top_k_seg_list[i], g_instance_top_k_view_mask_ids_list[i])
            avg_scores_t2s = torch.mean(view_scores_t2s)

            if avg_scores_s2t >= cp_score_threshold and avg_scores_t2s >= cp_score_threshold:
                g_instance_scores[g_instance_source_id][g_instance_target_id] = (avg_scores_s2t + avg_scores_t2s) / 2
                g_instance_scores[g_instance_target_id][g_instance_source_id] = (avg_scores_s2t + avg_scores_t2s) / 2
    # END for^2
    return g_instance_scores.bool()

