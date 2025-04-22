import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.ops import ball_query
import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm
from torch.utils.data import DataLoader

from tool.geometric_helpers import get_depth_mask, query_neighbors_mt, denoise, denoise2, crop_scene_points, connected_components_with_sizes
from tool.helper_functions import get_pointcloud, get_pointcloud_xyz, set_diagonal_to_zero
from voxel_hashing import VoxelHashTable


class VoxelBlockGrid:
    def __init__(self, cfg, dataset, device="cuda:0"):
        self.cfg = cfg
        self.dataset = dataset
        self.device = device
        self.device_o3c = o3c.Device(self.device)

        self.depth_far = self.cfg["cam"]["depth_far"]
        self.img_h = dataset.target_h
        self.img_w = dataset.target_w
        self.intrinsics = dataset.cam_intrinsic

        # params for o3d.VoxelBlockGrid
        self.voxel_size = self.cfg["scene"]["voxel_size"]  # default: 0.025, metrics: m
        self.block_resolution = self.cfg["scene"]["block_resolution"]  # default: 4
        self.block_count = self.cfg["scene"]["block_count"]  # max block num
        self.voxel_block_grids = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight', 'color'), (o3c.float32, o3c.float32, o3c.float32), ((1), (1), (3)),
            self.voxel_size, self.block_resolution, self.block_count, device=self.device_o3c
        )

        self.voxel_hash_table = VoxelHashTable(self, cfg, dataset, self.block_count, self.block_resolution, device)

        # params for seg
        self.denoise_rm_ratio = self.cfg["mask"]["denoise_rm_ratio"]  # default: 0.2
        self.few_pixel_threshold = self.cfg["mask"]["few_pixel_threshold"]
        self.few_voxel_threshold = self.cfg["mask"]["few_voxel_threshold"]
        self.voxel_on_recon_pc = self.cfg["mask"]["voxel_on_recon_pc"]
        self.coverage_threshold = self.cfg["mask"]["coverage_threshold"]
        self.query_radius = self.cfg["mask"]["query_radius"]
        # self.query_radius = 2 * self.voxel_size
    # END __init__()

    @property
    def get_max_voxel_num(self):
        return self.block_count * self.block_resolution**3

    # @brief: get so-far added block number
    @property
    def get_block_num(self):
        return self.voxel_block_grids.hashmap().size()

    # @brief: get Block coordinates of so-far collected all VoxelBlocks
    @property
    def get_all_block_coords(self):
        block_coords_all_o3c = self.voxel_block_grids.hashmap().key_tensor()[:self.voxel_block_grids.hashmap().size()]
        block_coords_all = torch.utils.dlpack.from_dlpack(block_coords_all_o3c.to_dlpack())
        return block_coords_all

    # @brief: Get indices(keys) of all so-far collected VoxelBlocks;
    #-@return: Tensor(block_num, )
    @property
    def get_all_block_indices(self):
        block_coords_all_o3c = self.voxel_block_grids.hashmap().key_tensor()[:self.voxel_block_grids.hashmap().size()]
        block_indices_all_o3c, _ = self.voxel_block_grids.hashmap().find(block_coords_all_o3c)
        block_indices_all = torch.utils.dlpack.from_dlpack(block_indices_all_o3c.to_dlpack())
        return block_indices_all

    # @brief: Get indices(keys) of all so-far collected voxels;
    #-@return: Tensor(block_num * block_res^3, )
    @property
    def get_all_voxel_indices(self):
        block_coords_all_o3c = self.voxel_block_grids.hashmap().key_tensor()[:self.voxel_block_grids.hashmap().size()]
        block_indices_all_o3c, _ = self.voxel_block_grids.hashmap().find(block_coords_all_o3c)

        _, voxel_indices_all_o3c = self.voxel_block_grids.voxel_coordinates_and_flattened_indices(block_indices_all_o3c)  # voxel coordinates & indices contained by given blocks
        voxel_indices_all = torch.utils.dlpack.from_dlpack(voxel_indices_all_o3c.to_dlpack())
        return voxel_indices_all

    # @brief: Get coordinates and indices(keys) of all so-far collected voxels;
    #-@return voxel_coords_all: Tensor(block_num * block_res^3, 3);
    #-@return voxel_indices_all: Tensor(block_num * block_res^3, ).
    @property
    def get_all_voxel_coords_indices(self):
        block_coords_all_o3c = self.voxel_block_grids.hashmap().key_tensor()[:self.voxel_block_grids.hashmap().size()]
        block_indices_all_o3c, _ = self.voxel_block_grids.hashmap().find(block_coords_all_o3c)

        voxel_coords_all_o3c, voxel_indices_all_o3c = self.voxel_block_grids.voxel_coordinates_and_flattened_indices(block_indices_all_o3c)  # voxel coordinates & indices contained by given blocks
        voxel_coords_all = torch.utils.dlpack.from_dlpack(voxel_coords_all_o3c.to_dlpack())
        voxel_indices_all = torch.utils.dlpack.from_dlpack(voxel_indices_all_o3c.to_dlpack())
        return voxel_coords_all, voxel_indices_all

    # @brief: transfer given Block coordinates to corresponding World coordinates;
    # @param block_coords: Tensor(n, 3), dtype=int32;
    # -@return: Tensor(n, 3), dtype=float32.
    def block_coords2world_coords(self, block_coords, change_to_o3c=False):
        block_world_coords = block_coords * (self.block_resolution * self.voxel_size)
        if change_to_o3c:
            block_world_coords = o3c.Tensor.from_numpy(block_world_coords.cpu().numpy()).to(device=self.device_o3c)
        return block_world_coords

    # @brief: transfer given Voxel coordinates(voxel's World coordinate) to Block coordinates;
    # @param voxel_coords: Tensor(n, 3), dtype=float32;
    #-@return: Tensor(n, 3).
    def voxel_coords2block_coords(self, voxel_coords, change_to_o3c=False):
        block_coords = torch.floor(voxel_coords / (self.block_resolution * self.voxel_size))
        block_coords = block_coords.to(torch.int32)
        if change_to_o3c:
            block_coords = o3c.Tensor.from_numpy(block_coords.cpu().numpy()).to(device=self.device_o3c)
        return block_coords

    # @brief: transfer given World coordinates to Voxel coordinates (voxel's World coordinate);
    # @param world_coords: Tensor(n, 3);
    #-@return:
    def world_coords2voxel_coords(self, world_coords, change_to_o3c=False):
        voxel_coord_indices = torch.floor(world_coords / self.voxel_size).long()
        voxel_coords = voxel_coord_indices.float() * self.voxel_size
        if change_to_o3c:
            voxel_coords = o3c.Tensor.from_numpy(voxel_coords.cpu().numpy()).to(device=self.device_o3c)
        return voxel_coords


    # @brief: (***) transfer given Voxel coordinates(voxel's World coordinate) to 1D voxel indices(which is also a voxel's key in voxel hash tabel);
    # @param voxel_coords: Tensor(n, 3), dtype=float32;
    #-@return: Tensor(n, ), dtype=int64.
    def voxel_coords2voxel_indices(self, voxel_coords, change_to_o3c=False):
        # Step 1: for each input voxel coordinate, get its corresponding VoxelBlock (its Block coordinates and 1D block index)
        corr_block_coords = self.voxel_coords2block_coords(voxel_coords)
        corr_block_coords_o3c = o3c.Tensor.from_numpy(corr_block_coords.cpu().numpy()).to(device=self.device_o3c)
        corr_block_indices_o3c, _ = self.voxel_block_grids.hashmap().find(corr_block_coords_o3c)  # Tensor(n, )

        corr_block_coords_w = self.block_coords2world_coords(corr_block_coords)  # corresponding blocks' World coordinates, Tensor(n, 3)
        corr_block_indices = torch.utils.dlpack.from_dlpack(corr_block_indices_o3c.to_dlpack()).to(torch.int64)  # corresponding blocks' 1D block indices, Tensor(n, )

        # Step 2: for each input voxel coordinate, compute its x/y/z offset inter its corresponding VoxelBlock
        inter_block_offsets = torch.round( (voxel_coords - corr_block_coords_w) / self.voxel_size ).to(torch.int64)  # Tensor(n, 3)
        indices_offsets = inter_block_offsets[:, 0] + inter_block_offsets[:, 1] * self.block_resolution + inter_block_offsets[:, 2] * self.block_resolution**2

        # Step 3: get voxel indices of each input voxel
        voxel_indices = corr_block_indices * self.block_resolution**3 + indices_offsets
        if change_to_o3c:
            voxel_indices = o3c.Tensor.from_numpy(voxel_indices.cpu().numpy()).to(device=self.device_o3c)
        return voxel_indices


    def get_mask_denoise_flag(self, seg_image, mask_id, min_component_size=1000):
        mask_seg_image = torch.where(seg_image == mask_id, torch.ones_like(seg_image), torch.zeros_like(seg_image))
        component_num, component_sizes = connected_components_with_sizes(mask_seg_image)

        valid_components_indices = [i for i in range(component_num) if component_sizes[i] > min_component_size]
        return len(valid_components_indices) > 1


    # @param mask_points: Tensor(n, 3);
    #-@return: Tensor(n', 3).
    def denoise_mask_pc(self, mask_points, voxel_down_sample=True, keep_max=True, rm_ratio=-1.):
        mask_pcd = o3d.geometry.PointCloud()
        mask_pcd.points = o3d.utility.Vector3dVector(mask_points.cpu().numpy())
        if voxel_down_sample:
            mask_pcd = mask_pcd.voxel_down_sample(voxel_size=self.voxel_size)

        if keep_max:
            mask_pcd, remain_index = denoise(mask_pcd, eps=self.query_radius)  # only keep max connected component
        else:
            denoise_rm_ratio = self.denoise_rm_ratio if rm_ratio < 0. else rm_ratio
            mask_pcd, remain_index = denoise2(mask_pcd, eps=self.query_radius, remove_percent=denoise_rm_ratio)  # only remove small clusters
            # mask_pcd, remain_index = denoise2(mask_pcd, eps=2 * self.voxel_size, remove_percent=denoise_rm_ratio)  # only remove small clusters

        mask_points_remained = torch.from_numpy(np.asarray(mask_pcd.points)).to(mask_points)
        return mask_points_remained


    # @brief: giving a segmented 2D mask image(and depth image), find the voxels corresponding to each mask;
    # @param depth_image: Tensor(H, W);
    # @param pose_c2w: Tensor(4, 4);
    # @param frame_voxel_coords: 该帧所包含的voxel的世界坐标, Tensor(v_num, 3), dtype=float32;
    # @param seg_image: segmentation mask image of this frame, each pixel corresponds to mask_id, Tensor(H, W), dtype=uint8, device=cuda;
    # @param mask_features: visual embedding of each mask in this frame, Tensor(mask_num, 512), dtype=float32;
    #-@return mask_info: voxel coordinates corresponding to each 2D mask, {mask_id: (voxel_coord0, voxel_coord1, voxel_coord2, ...)}, dict of Tensor, dtype=float32;
    #-@return valid_mask_ids: which 2D masks are valid in this frame, list(int);
    def turn_mask_to_voxel(self, frame_id, depth_image, pose_c2w, frame_voxel_coords, seg_image, mask_features=None):
        if torch.sum(torch.isinf(pose_c2w)) > 0:
            return {}, [], None, []

        # Step 1: preparation
        # 1.1: get all 2D mask_IDs
        seg_image_reshape = seg_image.reshape(-1)  # Tensor(H * W)
        mask_ids = torch.unique(seg_image_reshape)  # mask_ids of all detected masks, Tensor(m, ), dtype=uint8
        mask_ids.sort()

        # 1.2: compute per-pixel mask
        depth_mask = get_depth_mask(depth_image, dist_far=self.depth_far)  # Tensor(H * W), dtype=bool

        # 1.3: for each pixel in this RGB-D frame, compute its corresponding voxel
        frame_points = get_pointcloud_xyz(depth_image, self.intrinsics, pose_c2w, mask=depth_mask).float()  # pointcloud of this frame(in World CS), Tensor(pts_num, 3)

        # Step 2: for each 2D mask in current frame, compute its corresponding voxel coordinates in current frame's pointcloud
        valid_mask_ids = []
        valid_mask_point_list = []
        mask_points_num_list = []
        scene_points_list = []
        scene_points_num_list = []
        for mask_id in mask_ids:
            mask_id = mask_id.item()
            if mask_id == 0:
                continue

            # 2.1: extract corresponding points of this mask
            segmentation_mask = (seg_image_reshape == mask_id)  # 所有像素上mask_id等于给定mask_values的那些像素的mask, Tensor(H * W, ), dtype=bool
            valid_mask = segmentation_mask[depth_mask]  # 该帧上mask_id等于当前给定值 且 depth>0 的像素的mask, Tensor(pts_num, )
            if torch.count_nonzero(valid_mask) < self.few_pixel_threshold:
                continue
            mask_pts = frame_points[valid_mask]  # 该mask所对应的valid 3D points

            # 2.2: 去noise, 得到该mask对应的voxels centers在世界坐标系中的位置
            # multi_component_flag = self.get_mask_denoise_flag(seg_image, mask_id)
            mask_pts_remained = self.denoise_mask_pc(mask_pts, keep_max=True)
            if mask_pts_remained.shape[0] < self.few_voxel_threshold:
                continue

            cropped_scene_points, selected_point_ids = crop_scene_points(mask_pts_remained, frame_voxel_coords, pad_dist=self.query_radius)
            valid_mask_ids.append(mask_id)
            valid_mask_point_list.append(mask_pts_remained)
            mask_points_num_list.append(mask_pts_remained.shape[0])
            scene_points_list.append(cropped_scene_points)
            scene_points_num_list.append(cropped_scene_points.shape[0])


        # Step 3: for each valid mask, get its corresponding voxels by ball_query
        if len(valid_mask_ids) == 0 or all( [sp.numel() == 0 for sp in scene_points_list] ):
            return {}, [], None, []
        mask_points_tensor = pad_sequence(valid_mask_point_list, batch_first=True, padding_value=0)
        cropped_scene_pts_tensor = pad_sequence(scene_points_list, batch_first=True, padding_value=0)

        lengths_1 = torch.tensor(mask_points_num_list).to(self.device)
        lengths_2 = torch.tensor(scene_points_num_list).to(self.device)
        corr_voxels_in_scene = query_neighbors_mt(mask_points_tensor, cropped_scene_pts_tensor, lengths_1, lengths_2, radius=self.query_radius)  # neighbor point_ids in scene_points for each mask pointcloud, Tensor(v_mask_num, mask_pts_num, K)

        # Step 4: record corresponding voxels(their 3D coordinates in World CS) for each valid mask
        valid_mask_num = len(valid_mask_ids)
        final_valid_mask_ids = []
        valid_mask_features = []
        mask_info = {}
        frame_voxels_coords = []
        for i in range(valid_mask_num):
            mask_id = valid_mask_ids[i]

            # 4.1: get query result for this mask
            mask_neighbor = corr_voxels_in_scene[i]
            mask_point_num = mask_points_num_list[i]  # Pi
            mask_neighbor = mask_neighbor[:mask_point_num]  # Tensor(Pi, 10)

            valid_neighbor = (mask_neighbor != -1)  # Tensor(Pi, 10), dtype=bool
            neighbor = torch.unique(mask_neighbor[valid_neighbor])
            corr_involved_voxels = scene_points_list[i][neighbor]
            coverage = torch.any(valid_neighbor, dim=1).sum().item() / mask_point_num

            if (coverage < self.coverage_threshold) or (corr_involved_voxels.shape[0] < self.few_voxel_threshold):
                continue

            final_valid_mask_ids.append(mask_id)
            if mask_features is not None:
                valid_mask_features.append(mask_features[mask_id - 1])
            mask_info[mask_id] = corr_involved_voxels
            frame_voxels_coords.append(corr_involved_voxels)

        if len(frame_voxels_coords) > 0:
            frame_voxels_coords = torch.cat(frame_voxels_coords, dim=0)
            uniq_frame_voxel_coords = torch.unique(frame_voxels_coords, dim=0)
        else:
            uniq_frame_voxel_coords = None

        return mask_info, final_valid_mask_ids, uniq_frame_voxel_coords, valid_mask_features

