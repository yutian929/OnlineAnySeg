import copy
import os
import numpy as np
import torch


# @brief: make RGB color lighter
def adjust_colors_to_pastel(rgb_array, factor=0.9):
    pastel_rgb_array = rgb_array * factor + (1 - factor)
    return pastel_rgb_array


def vis_one_object(point_ids, scene_points):
    points = scene_points[point_ids]
    color = (torch.rand(3) * 0.7 + 0.3) * 255
    colors = torch.tile(color, (points.shape[0], 1))
    pts_mean = torch.mean(points, dim=0)
    return point_ids, points, colors, color, pts_mean

def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        label = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (label > 0):
            pallete[j * 3 + 0] |= (((label >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((label >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((label >> 2) & 1) << (7 - i))
            i = i + 1
            label >>= 3

    # set first instance always be yellow
    pallete[0] = 255
    pallete[1] = 215
    pallete[2] = 0
    return pallete

def generate_distinct_colors(n, seed=15):
    torch.manual_seed(seed)

    colors = torch.empty((n, 3))
    colors[0] = torch.rand(3)
    for i in range(1, n):
        # 随机生成候选颜色
        candidate_colors = torch.rand(100, 3)
        distances = torch.cdist(colors[:i], candidate_colors)
        min_distances, _ = distances.min(dim=0)
        best_color_idx = min_distances.argmax()
        colors[i] = candidate_colors[best_color_idx]

    colors = adjust_colors_to_pastel(colors)
    return colors

