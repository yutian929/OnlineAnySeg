import numpy as np
import torch


class Instance:
    def __init__(self, frame_id, merged_mask_id, mask_voxel_coords, mask_voxel_indices, ori_mask_list, mask_sem_feature, cfg, rgb=None, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        self.rgb = rgb

        self.frame_id = frame_id
        self.mask_id = merged_mask_id
        self.mask_voxel_coords = mask_voxel_coords
        self.mask_voxel_indices = mask_voxel_indices
        self.ori_mask_list = ori_mask_list  # list of (frame_ID, mask_ID)
        self.mask_sem_feature = mask_sem_feature
        self.visible_frame_ids = [ori_mask[0] for ori_mask in ori_mask_list]  # list of frame_ID

        if len(self.ori_mask_list) > 1:
            self.ori_mask_list.sort()
        if len(self.visible_frame_ids) > 1:
            self.visible_frame_ids = sorted(set(self.visible_frame_ids))
    # END __init__()


    @staticmethod
    def creat_instance_from_list(frame_id, merged_mask_id, instance_list, cfg, device="cuda:0"):
        instance_list = sorted(instance_list, key=lambda x: x.mask_voxel_coords.shape[0], reverse=True)  # sort all combined instances in descending order
        rgb_this = instance_list[0].rgb

        ori_mask_list_new = []
        mask_sem_feature_list = []
        mask_voxel_coords_list = []
        mask_voxel_indices_list = []
        for instance in instance_list:
            ori_mask_list_new += instance.ori_mask_list
            mask_sem_feature_list.append(instance.mask_sem_feature)
            mask_voxel_coords_list.append(instance.mask_voxel_coords)
            mask_voxel_indices_list.append(instance.mask_voxel_indices)
        ori_mask_list_new = set(ori_mask_list_new)

        mask_voxel_coords_new = torch.concat(mask_voxel_coords_list, dim=0)
        mask_voxel_coords_new = torch.unique(mask_voxel_coords_new, dim=0)

        mask_voxel_indices_new = torch.concat(mask_voxel_indices_list, dim=0)
        mask_voxel_indices_new = torch.unique(mask_voxel_indices_new, dim=0)

        mask_sem_feature_new = torch.stack(mask_sem_feature_list, dim=0).sum(dim=0)  # Tensor(feature_dim, )
        new_instance = Instance(frame_id, merged_mask_id, mask_voxel_coords_new, mask_voxel_indices_new, list(ori_mask_list_new), mask_sem_feature_new, cfg, rgb_this, device)
        return new_instance

    @property
    def get_semantic_feature(self):
        sem_feature_mat = self.mask_sem_feature / (self.mask_sem_feature.norm(dim=-1, keepdim=True) + 1e-7)  # normalization for extracted visual embeddings
        return sem_feature_mat



