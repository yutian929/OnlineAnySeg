import torch


# @brief: sample pixels uniformly from a frame;
#-@return rows: row_Id of sampled pixels, Tensor(num_h * num_w), dtype=torch.int64;
#-@return cols: col_Id of sampled pixels, Tensor(num_h * num_w), dtype=torch.int64.
def sample_pixels_uniformly(img_h, img_w, num_h, num_w):
    interval_h, offset_h = (img_h - num_h) // (num_h + 1), (img_h - num_h) % (num_h + 1)
    interval_w, offset_w = (img_w - num_w) // (num_w + 1), (img_w - num_w) % (num_w + 1)

    row_Ids = torch.arange(0, num_h, dtype=torch.int64) * (interval_h + 1) + interval_h + offset_h // 2  # Tensor(num_h, )
    col_Ids = torch.arange(0, num_w, dtype=torch.int64) * (interval_w + 1) + interval_w + offset_w // 2  # Tensor(num_w, )

    rows = row_Ids[..., None].repeat((1, num_w)).reshape((-1, ))  # Tensor(num_h, num_w)
    cols = col_Ids[None, ...].repeat((num_h, 1)).reshape((-1, ))  # Tensor(num_h, num_w)
    return rows, cols


def pixel_rc_to_indices(H, W, rows, cols):
    indices = rows * W + cols
    return indices


def compute_pixel_mask(img_h, img_w, downsample_ratio_h, downsample_ratio_w, device="cpu"):
    num_h = int(img_h // downsample_ratio_h)
    num_w = int(img_w // downsample_ratio_w)
    rows, cols = sample_pixels_uniformly(img_h, img_w, num_h, num_w)
    pixel_indices = pixel_rc_to_indices(img_h, img_w, rows, cols)

    pixel_mask = torch.zeros((img_h * img_w,), dtype=torch.bool, device=device)
    pixel_mask[pixel_indices] = True
    return pixel_mask

