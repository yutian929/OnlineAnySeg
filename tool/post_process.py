import os
import numpy as np
import torch
import open3d as o3d
from sklearn.cluster import DBSCAN


# @brief: filter instances
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
        dbscan.fit(pts)

        unique_labels, counts = np.unique(dbscan.labels_, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]

        keeping_mask = (dbscan.labels_ == largest_cluster_label)
        inlier_indices = pt_indices[keeping_mask]
        instance_mask_new = np.zeros_like(instance_mask)
        instance_mask_new[inlier_indices] = True

        final_pts = pts[keeping_mask]
        final_pts_list.append(final_pts)

        if len(inlier_indices) > size_thresh:
            instance_mask_list_new.append(instance_mask_new)
            instance_feature_list_new.append(instance_feature_list[i])
            valid_instance_indices.append(i)

    return instance_mask_list_new, instance_feature_list_new, valid_instance_indices


# @brief: save class-agnostic masks of this sequence;
def export_instance_mask(output_path, class_agnostic_mask_list, instance_sem_features):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_instance = len(class_agnostic_mask_list)
    if num_instance == 0:
        print("No Segmented instances result needs to be saved to: %s" % output_path)
        return

    pred_masks = np.stack(class_agnostic_mask_list, axis=1)  # ndarray(point_num, instance_num), dtype=bool
    pred_sem_features = np.stack(instance_sem_features, axis=1)
    pred_dict = {
        "pred_masks": pred_masks,  # ndarray(point_num, instance_num)
        "pred_score":  np.ones(num_instance),  # ndarray(instance_num, ), dtype=float64
        "pred_classes": np.zeros(num_instance, dtype=np.int32),  # ndarray(instance_num, ), dtype=int32
        "pred_sem_features": pred_sem_features,
    }

    np.savez(output_path, **pred_dict)
    print("Segmented instances result was saved to: %s" % output_path)

