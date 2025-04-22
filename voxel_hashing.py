import time
import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm
from torch.utils.data import DataLoader

from tool.helper_functions import get_pointcloud, get_pointcloud_xyz, merge_sets_and_count


class VoxelHashTable:
    def __init__(self, voxel_block_grid, cfg, dataset, block_size, block_resolution, device="cuda:0"):
        self.cfg = cfg
        self.dataset = dataset
        self.device = device
        self.device_o3c = o3c.Device(self.device)

        self.voxel_size = self.cfg["scene"]["voxel_size"]  # default: 0.025, metrics: m
        self.max_block_num = block_size
        self.block_resolution = block_resolution
        self.max_voxel_num = self.max_block_num * self.block_resolution**3  # max voxel number (pre-defined)
        self.voxel_grids = voxel_block_grid
        self.voxel_block_grids = self.voxel_grids.voxel_block_grids

        self.hash_table = None
        self.hash_item_flag = None
        self.initialize()

    @property
    def get_occupied_voxels(self):
        return torch.where(self.hash_item_flag)[0]

    def initialize(self):
        self.hash_table = [None for _ in range(self.max_voxel_num)]
        self.hash_item_flag = torch.zeros((self.max_voxel_num, ), dtype=torch.bool, device=self.device)  # whether each voxel is occupied, Tensor(voxel_num, ), dtype=bool

    # @brief: transfer given Block coordinates to corresponding World coordinates;
    # @param block_coords: Tensor(n, 3), dtype=int32;
    #-@return: Tensor(n, 3), dtype=float32.
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


    # @brief: Hash function - 计算给定3D voxel坐标的
    # @param voxel_coords: n个给定点的3D Voxel坐标, Tensor(n, 3), dtype=int64;
    #-@return: 这n个坐标各自对应的hash key, Tensor(n, ), dtype=int64.
    def get_hash_keys(self, voxel_coords):
        pts_key = self.voxel_coords2voxel_indices(voxel_coords, change_to_o3c=False)
        return pts_key

    # @brief: insert a new mask --- add its global_mask_id to all its related voxels
    # @param voxel_coords: Voxel coordinates(in World CS), Tensor(n, 3), dtype=float32;
    # @param global_mask_id: 1D global_ID of an original 2D mask, int;
    def insert_mask_voxels(self, voxel_coords, global_mask_id):
        voxel_ids = self.get_hash_keys(voxel_coords)  # Tensor(n, ), dtype=int64
        voxel_ids = voxel_ids.tolist()

        for voxel_id in voxel_ids:
            if self.hash_table[voxel_id] is None:
                self.hash_table[voxel_id] = set()
            self.hash_table[voxel_id].add(global_mask_id)

    def remove_mask_voxels(self, voxel_coords, global_mask_id):
        voxel_ids = self.get_hash_keys(voxel_coords)  # Tensor(n, ), dtype=int64
        voxel_ids = voxel_ids.tolist()

    # @brief: for a batch of given voxels, compute overlap mask_ids and their overlap count;
    # @param voxel_coords: Voxel coordinates(in World CS), Tensor(n, 3), dtype=float32;
    #-@return overlap_masks: global Mask_ID(1D ID for each raw masks) of each overlapped mask, Tensor(ovlp_mask_num, ), dtype=int64;
    #-@return mask_counts: overlap voxel number between query voxel set and each of overlapped mask, Tensor(ovlp_mask_num, ), dtype=int64.
    def query_mask(self, voxel_coords):
        # Step 1: get items in hash table of each given Voxel coordinates
        voxel_ids = self.get_hash_keys(voxel_coords)  # Tensor(n, )
        voxel_ids = voxel_ids.tolist()

        voxel_items = []  # list of set
        for voxel_id in voxel_ids:
            voxel_global_mask_ids = self.hash_table[voxel_id]  # global_mask_ids that this voxel corresponds to, set / None
            if voxel_global_mask_ids is not None:
                voxel_merged_mask_ids = {global_mask_id for global_mask_id in voxel_global_mask_ids}  # set of global_mask_id this voxel corresponds to
            else:
                voxel_merged_mask_ids = None
            voxel_items.append(voxel_merged_mask_ids)

        # Step 2:
        overlap_masks, mask_counts = merge_sets_and_count(voxel_items, self.device)
        return overlap_masks, mask_counts


    # @brief: for a batch of given voxels, compute overlap mask_ids and their overlap count;
    # @param voxel_coords: Voxel coordinates(in World CS), Tensor(n, 3), dtype=float32;
    # @param id_mapping: mapping global_mask_id to merged_mask_id, list of int;
    # @param valid_merge_mask_ids: merge_mask_ids to count, list/None;
    def query_mask_w_mapping(self, voxel_coords, id_mapping, valid_merge_mask_ids=None):
        mapping_list_len = len(id_mapping)

        # Step 1: get items in hash table of each given Voxel coordinates
        voxel_ids = self.get_hash_keys(voxel_coords)  # Tensor(n, )
        voxel_ids = voxel_ids.tolist()

        voxel_items = []  # list of set
        for voxel_id in voxel_ids:
            voxel_global_mask_ids = self.hash_table[voxel_id]  # global_mask_ids that this voxel corresponds to, set / None
            if voxel_global_mask_ids is not None:
                if valid_merge_mask_ids is None:
                    voxel_merged_mask_ids = {id_mapping[global_mask_id] for global_mask_id in voxel_global_mask_ids if global_mask_id < mapping_list_len}  # set of merged_mask_id this voxel corresponds to
                else:
                    voxel_merged_mask_ids = {id_mapping[global_mask_id] for global_mask_id in voxel_global_mask_ids if
                                             (global_mask_id < mapping_list_len and id_mapping[global_mask_id] in valid_merge_mask_ids)}  # set of merged_mask_id this voxel corresponds to
            else:
                voxel_merged_mask_ids = None
            voxel_items.append(voxel_merged_mask_ids)

        # Step 2: for each overlap mask, count its overlap voxel number
        overlap_mask_ids, overlap_mask_counts = merge_sets_and_count(voxel_items, self.device)

        return overlap_mask_ids, overlap_mask_counts, voxel_ids
