import numpy as np
import open3d as o3d
import torch
from natsort import natsorted

from tool.visualization_helpers import vis_pcd


def highlight_points(points, mask):
    # colors = np.asarray(pcd.colors)
    # colors = np.zeros_like(colors)
    # colors[mask, :] = np.array([1., 0., 0.])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color((0, 0, 0))

    vis_colors = np.asarray(pcd.colors)
    vis_colors[mask] = (1., 0., 0.)
    vis_pcd(pcd)

def get_all_seq(seq_file, ignore_sub_seq=False):
    with open(seq_file, "r", encoding='utf-8') as f_in:
        seq_list = f_in.readlines()
    seq_list = [seq_name.strip() for seq_name in seq_list]
    seq_list = natsorted(seq_list)

    if ignore_sub_seq:
        seq_dict = {}
        for seq_name in seq_list:
            seq_category_name = seq_name.split("_")[0]
            if seq_category_name not in seq_dict:
                seq_dict[seq_category_name] = []
            seq_dict[seq_category_name].append(seq_name)

        seq_list = [seq_names[0] for seq_category_name, seq_names in seq_dict.items()]

    return seq_list
