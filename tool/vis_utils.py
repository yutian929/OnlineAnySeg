import numpy as np
import torch
import open3d as o3d

from tool.visualization_helpers import get_new_pallete


class Vis_color:
    def __init__(self, use_vis):
        self.use_vis = use_vis
        if not self.use_vis:
            return None
        self.vis_image = o3d.visualization.Visualizer()
        self.vis_image.create_window(window_name="input color image", width=320, height=240, left=1750)

        self.pallete = get_new_pallete(100)  # list(3 * cls_num)

    # @param color_image: Tensor(H, W, 3), dtype=float32
    def update(self, color_image):
        color_img_nd = (color_image.cpu().numpy() * 255).astype(np.uint8)

        if not self.use_vis:
            return
        geometry_image = o3d.geometry.Image(color_img_nd)
        self.vis_image.add_geometry(geometry_image)
        self.vis_image.poll_events()
        self.vis_image.update_renderer()
        geometry_image.clear()


class Vis_pointcloud:
    def __init__(self, use_vis, args, device="cuda:0"):
        self.use_vis = use_vis
        self.args = args
        self.device = device
        if not self.use_vis:
            return None

        self.text_embeddings = self.get_text_embeddings().to(self.device)  # Tensor(label_num, feature_dim), dtype=float32
        self.label_num = self.text_embeddings.shape[0]
        self.pallete = self.get_color_pallete(self.label_num)  # ndarray(label_num, 3), dtype=int64

        self.add_geo_flag = False

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="segmentation", width=1460, height=720, left=250)
        render_option = self.vis.get_render_option()
        render_option.point_size = 2.7

        self.pcd = o3d.geometry.PointCloud()
        self.scene_points = None
        self.scene_points_color = None

        self.cam = None
        ctr = self.vis.get_view_control()
        ctr.set_constant_z_far(1000)

    def get_text_embeddings(self):
        text_embed_path = self.args.vocab_feature_file
        text_embeddings = torch.load(text_embed_path)
        return text_embeddings

    def get_color_pallete(self, label_num, brighten=True):
        pallete = get_new_pallete(label_num)  # list(3 * label_num)
        pallete = np.array(pallete).reshape((-1, 3))  # RGB value of each label, [0~255], ndarray(label_num, 3), dtype=int64

        if brighten:
            pallete_float = pallete.astype("float32") / 255.
            pallete_float = np.power(pallete_float, 1 / 2.2)
            pallete = (pallete_float * 255).astype("int64")
        return pallete

    def set_uniform_color(self, scene_colors, rgb=[156, 156, 156]):
        rgb = np.array(rgb)
        scene_colors[:] = rgb
        return scene_colors

    # @brief: update members "self.scene_points" and "self.scene_points_color" to latest reconstruction and segmentation results;
    # @param points: ndarray(scene_pts_num, 3);
    # @param instance_list: list of Instance obj;
    # @param instance_pts_mask_list: list of Tensor(scene_pts_num, ), dtype=bool;
    def show_current_seg_pc(self, scene_points, instance_list, instance_pts_mask_list):
        num_instances = len(instance_list)

        # Step 1: get each instance's most possible category and its corresponding RGB value
        inst_features = [instance.get_semantic_feature for instance in instance_list]
        inst_features = torch.stack(inst_features, dim=0)

        inst_label_score = inst_features @ self.text_embeddings.T  # Tensor(instance_num, label_num)
        max_scores, max_label_ids = torch.max(inst_label_score, dim=-1)
        max_label_ids = max_label_ids.cpu().numpy()
        instance_label_rgb = self.pallete[max_label_ids]  # RGB value(0~255) of each instance, ndarray(instance_num, 3), dtype=int64

        # Step 2: for each detected valid mask, compute its corresponding points in scene points
        scene_colors = np.zeros_like(scene_points).astype("int64")
        scene_colors = self.set_uniform_color(scene_colors)

        for idx in range(num_instances):
            instance_mask = instance_pts_mask_list[idx].cpu().numpy()
            corr_point_ids = np.where(instance_mask)[0]
            scene_colors[corr_point_ids] = instance_label_rgb[idx]

        self.scene_points = scene_points.astype("float64")
        self.scene_points_color = scene_colors.astype("float64")


    # @brief: show seg pc from 0 to T;
    # @param points: points to show, ndarray(n, 3), dtype=float64;
    # @param points_color: RGB(0~255) of points to show, ndarray(n, 3), dtype=float64;
    def update(self):
        if not self.use_vis:
            return

        if self.scene_points is not None and self.scene_points_color is not None:
            self.pcd.points = o3d.utility.Vector3dVector(self.scene_points)
            self.pcd.colors = o3d.utility.Vector3dVector(self.scene_points_color / 255)
        else:
            return

        if not self.add_geo_flag:
            self.vis.add_geometry(self.pcd)
            self.add_geo_flag = True
        else:
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()
