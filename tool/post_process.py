import os
import numpy as np
import torch
import open3d as o3d
from sklearn.cluster import DBSCAN


# @brief: 对于每个3D instance, 用DBSCAN找出其最大cluster作为其filter后的instance; 然后剔除那些filtered instance点数较少的instance;
# @param scene_pcd: 场景的GT点云, o3d.PointCloud obj;
# @param instance_pt_count: 每个existing instance中的每个点分别对应多少个2D instance, Tensor(e_instance_num, pts_num);
# @param instance_features: 每个existing instance的CLIP feature, Tensor(e_instance_num, 512);
# @param size_thresh
#-@return instance_pt_count:
#-@return instance_features:
def filter_by_clustering(scene_pcd, num_instances, instance_pt_count, instance_pt_score, instance_pt_mask, instance_features, size_thresh=200):
    # ===================== filter by segment connectivity =====================
    pcd_pts = []  # 记录每个3D instance的点云
    keep_indices = []

    for i in range(num_instances):
        pt_indices = instance_pt_count[i].nonzero()[:, 0].detach().cpu().numpy()  # 该3D instance的所有点的point_idx
        segment_pcd = scene_pcd.select_by_index(pt_indices)
        pts = np.array(segment_pcd.points)
        pcd_pts.append(pts)

        dbscan = DBSCAN(eps=0.1, min_samples=1)
        dbscan.fit(pts)  # perform DBSCAN对该instance上的所有3D点进行聚类

        unique_labels, counts = np.unique(dbscan.labels_, return_counts=True)
        largest_cluster_label = unique_labels[ np.argmax(counts) ]  # 最大cluster对应的label
        inlier_indices = pt_indices[dbscan.labels_==largest_cluster_label]  # 最大cluster对应的点(在所有点中的)indices
        outlier_mask = np.ones(instance_pt_count.shape[1], dtype=bool)
        outlier_mask[inlier_indices] = False  # 该instance中inlier点为False, outlier点为True
        instance_pt_count[i, outlier_mask] = 0
        if len(inlier_indices) > size_thresh:
            keep_indices.append(i)

    keep_indices = torch.LongTensor(keep_indices)
    instance_pt_count = instance_pt_count[keep_indices]
    instance_pt_score = instance_pt_score[keep_indices]
    instance_pt_mask = instance_pt_mask[keep_indices]
    instance_features = instance_features[keep_indices]

    return instance_pt_count, instance_pt_score, instance_pt_mask, instance_features

# @brief:
# @param recon_pc: reconstructed pointcloud, ndarray(pc_num, 3);
# @param instance_mask_list: each instance's point mask in given pc, list of ndarray(pc_num, ), dtype=bool;
def filter_instances(recon_pc, instance_mask_list, instance_feature_list, size_thresh=200, device="cuda:0"):
    num_instances = len(instance_mask_list)
    instance_mask_list_new = []
    instance_feature_list_new = []
    valid_instance_indices = []

    raw_pts_list = []
    final_pts_list = []
    for i in range(num_instances):
        instance_mask = instance_mask_list[i]
        pt_indices = np.where(instance_mask)[0]
        pts = recon_pc[pt_indices]
        raw_pts_list.append(pts)  # TEST

        dbscan = DBSCAN(eps=0.1, min_samples=1)
        dbscan.fit(pts)  # perform DBSCAN对该instance上的所有3D点进行聚类

        unique_labels, counts = np.unique(dbscan.labels_, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]  # 最大cluster对应的label

        keeping_mask = (dbscan.labels_ == largest_cluster_label)
        inlier_indices = pt_indices[keeping_mask]  # 最大cluster对应的点(在所有点中的)indices
        instance_mask_new = np.zeros_like(instance_mask)
        instance_mask_new[inlier_indices] = True

        final_pts = pts[keeping_mask]
        final_pts_list.append(final_pts)

        if len(inlier_indices) > size_thresh:
            instance_mask_list_new.append(instance_mask_new)
            instance_feature_list_new.append(instance_feature_list[i])
            valid_instance_indices.append(i)

    # ############################### TEST ###############################
    # save_pts_list("/home/javens/git_repos/LG_seg_20240826/output/scene0015_00/pred_instance_raw", raw_pts_list)
    # save_pts_list("/home/javens/git_repos/LG_seg_20240826/output/scene0015_00/pred_instance_filtered", final_pts_list, pc_color=[0., 0., 1.])
    # ############################### END TEST ###############################

    return instance_mask_list_new, instance_feature_list_new, valid_instance_indices



# @brief: 保存该sequence的class-agnostic masks;
# @param class_agnostic_mask_list: 最终分割出来的每个3D mask在GT pointcloud上对应的mask, list of ndarray(point_num, ), dtype=bool;
def export_instance_mask(output_path, class_agnostic_mask_list, instance_sem_features):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_instance = len(class_agnostic_mask_list)
    if num_instance == 0:
        print("No Segmented instances result needs to be saved to: %s" % output_path)
        return

    pred_masks = np.stack(class_agnostic_mask_list, axis=1)  # ndarray(point_num, instance_num), dtype=bool
    pred_sem_features = np.stack(instance_sem_features, axis=1)
    pred_dict = {
        "pred_masks": pred_masks,  # 该seq的GT点云上各点是否属于各detected 3D instance的mask, ndarray(point_num, instance_num)
        "pred_score":  np.ones(num_instance),  # ndarray(instance_num, ), dtype=float64
        "pred_classes": np.zeros(num_instance, dtype=np.int32),  # ndarray(instance_num, ), dtype=int32
        "pred_sem_features": pred_sem_features,
    }

    np.savez(output_path, **pred_dict)
    print("Segmented instances result was saved to: %s" % output_path)


########################################### helper functions ###########################################
def save_pts_list(output_dir, pts_list, mask_ids=None, pc_color=[1., 0., 0.]):
    os.makedirs(output_dir, exist_ok=True)
    for i, instance_pts in enumerate(pts_list):
        if mask_ids is not None:
            mask_id = mask_ids[i]
        else:
            mask_id = i
        output_path = os.path.join(output_dir, f"{mask_id}.ply")

        if isinstance(instance_pts, torch.Tensor):
            instance_pts = instance_pts.cpu().numpy()

        points = instance_pts.astype("float64")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        pcd.paint_uniform_color(np.array(pc_color).astype("float64"))

        o3d.io.write_point_cloud(output_path, pcd)

