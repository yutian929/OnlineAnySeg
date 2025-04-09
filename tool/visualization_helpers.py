import copy
import os
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# @brief: 将 RGB 颜色值调整为更淡的颜色（比如淡红、淡黄等）
def adjust_colors_to_pastel(rgb_array, factor=0.9):
    pastel_rgb_array = rgb_array * factor + (1 - factor)
    return pastel_rgb_array


# @brief: 给定完整场景的点云和某个instance对应的point_IDs, 返回该instance的子点云和颜色点云
# @param point_ids: Tensor(m, )
# @param scene_points: Tensor(n, 3)
#-@return point_ids
#-@return points
#-@return colors
#-@return color
#-@return pts_mean
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


def get_new_mask_pallete(new_palette, labels, out_label_flag=False):
    """Get image color pallete for visualizing masks"""
    labels_uniq = []
    for label in labels:
        if label not in labels_uniq:
            labels_uniq.append(label)

    patches = []
    for index, label in enumerate(labels_uniq):
        cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0,
                     new_palette[index * 3 + 2] / 255.0]
        red_patch = mpatches.Patch(color=cur_color, label=label)
        patches.append(red_patch)
    return patches


# @brief:
# @param n:
# @param seed: random seed;
#-@return: Tensor(n, 3)
def generate_distinct_colors(n, seed=15):
    torch.manual_seed(seed)  # 设置随机种子

    # 初始化第一个颜色
    colors = torch.empty((n, 3))  # 预分配 RGB 容器
    colors[0] = torch.rand(3)  # 随机生成第一个颜色

    for i in range(1, n):
        # 随机生成候选颜色
        candidate_colors = torch.rand(100, 3)  # 每次生成 100 个候选颜色
        distances = torch.cdist(colors[:i], candidate_colors)  # 计算已有颜色与候选颜色的距离
        min_distances, _ = distances.min(dim=0)  # 找到每个候选颜色的最小距离
        best_color_idx = min_distances.argmax()  # 选择最远的候选颜色
        colors[i] = candidate_colors[best_color_idx]  # 添加到颜色集合中

    colors = adjust_colors_to_pastel(colors)
    return colors


def visualize_colors(colors, output_path="colors.png"):
    """
    将生成的颜色可视化为矩阵图，并保存为图片。

    Args:
        colors (torch.Tensor): n x 3 的 RGB 颜色张量，范围为 [0, 1]。
        output_path (str): 输出图片的保存路径。
    """
    n = colors.shape[0]  # 颜色数量
    matrix_size = int(torch.ceil(torch.sqrt(torch.tensor(n)).float()).item())  # 确定矩阵的尺寸
    matrix = np.zeros((matrix_size, matrix_size, 3))  # 初始化矩阵

    # 将生成的颜色填入矩阵
    for i in range(n):
        row, col = divmod(i, matrix_size)
        matrix[row, col] = colors[i].numpy()

    # 可视化矩阵
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix)
    plt.axis('off')  # 关闭坐标轴
    plt.title("Visualized Colors", fontsize=16)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved color visualization to {output_path}")


def vis_pcd(pcd, cam_pose=None, coord_frame_size=0.2):
    if not isinstance(pcd, list):
        pcd = [pcd]
    pcd_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=coord_frame_size)
    if cam_pose is not None:
        cam_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=coord_frame_size)
        cam_frame.transform(cam_pose)
        o3d.visualization.draw_geometries([*pcd, pcd_frame, cam_frame])
    else:
        o3d.visualization.draw_geometries([*pcd, pcd_frame])


# @brief:
# @param scene_pcd: Gt pointcloud;
# @param scene_graph: scene graph, 其中每个node代表一个instance.
def visualize_scene_graph(scene_pcd, scene_graph, show_center=False, legend_output="./legend_test.png", pc_output_dir="./output", pc_output_file="pc_segmented.ply"):
    geometries = []
    pcd_vis = copy.deepcopy(scene_pcd)
    vis_colors = np.asarray(pcd_vis.colors)
    vis_colors[:] = (0, 0, 0)
    colors = np.random.random((len(scene_graph.nodes), 3))

    node_ids = list(scene_graph.nodes)
    print(f"{len(node_ids) = }")
    node_ids.sort(key=lambda x: len(scene_graph.nodes[x]["pt_indices"]), reverse=True)

    # Step 1: 绘制图例
    instance_labels = []
    node_label_ids = []
    for node_id in node_ids:
        instance_label = scene_graph.nodes[node_id]["top5_vocabs"][0]
        if instance_label not in instance_labels:
            instance_labels.append(instance_label)
        node_label_id = instance_labels.index(instance_label)
        node_label_ids.append(node_label_id)

    pallete = get_new_pallete(len(instance_labels))  # list of int (3 * label_num)
    patches = get_new_mask_pallete(pallete, instance_labels)
    colors = np.array(pallete).reshape((-1, 3)) / 255.0

    plt.figure(figsize=(8, 6))
    ax2 = plt.subplot(1, 1, 1)
    ax2.axis("off")
    ax2.legend(handles=patches, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 15})
    plt.savefig(legend_output)  # 保存plt(也就是保存该image)

    # Step 2: 给原点云中的每个instance上色
    # naive visualization (sort by segment size and visualize large instances first)
    for i, node_id in enumerate(node_ids):
        node = scene_graph.nodes[node_id]
        pt_indices = node["pt_indices"]
        # vis_colors[pt_indices] = colors[node_id]
        label_id = node_label_ids[i]
        vis_colors[pt_indices] = colors[label_id]
        label = instance_labels[label_id]
        if show_center:
            mesh_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.02)
            mesh_sphere.compute_vertex_normals()
            mesh_sphere.paint_uniform_color((1, 0, 0))
            mesh_sphere.translate(node["center"])
            geometries.append(mesh_sphere)
            instance_output_path = os.path.join(pc_output_dir, "%d_%s.ply" % (i, label))
            o3d.io.write_triangle_mesh(instance_output_path, mesh_sphere)
    
    geometries.append(pcd_vis)
    # o3d.io.write_point_cloud(os.path.join(pc_output_dir, "pc_original.ply"), scene_pcd)
    o3d.io.write_point_cloud(os.path.join(pc_output_dir, pc_output_file), pcd_vis)
    vis_pcd(geometries)
