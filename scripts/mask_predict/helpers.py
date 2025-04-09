import os
import cv2
import numpy as np
import torch
from PIL import Image


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


def mask2box_multi_level(mask, level, expansion_ratio):
    pos = np.where(mask)
    top = np.min(pos[0])
    bottom = np.max(pos[0])
    left = np.min(pos[1])
    right = np.max(pos[1])

    if level == 0:
        return left, top, right, bottom
    shape = mask.shape
    x_exp = int(abs(right - left) * expansion_ratio) * level
    y_exp = int(abs(bottom - top) * expansion_ratio) * level
    return max(0, left - x_exp), max(0, top - y_exp), min(shape[1], right + x_exp), min(shape[0], bottom + y_exp)


def crop_image(rgb, mask, CROP_SCALES=3):
    multiscale_cropped_images = []
    for level in range(CROP_SCALES):
        left, top, right, bottom = mask2box_multi_level(mask, level, 0.1)
        cropped_image = rgb[top:bottom, left:right].copy()
        multiscale_cropped_images.append(cropped_image)
    return multiscale_cropped_images


def get_cropped_image(mask, rgb):
    mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)  # resize mask image to the same size with input RGB image
    multiscale_cropped_images = crop_image(rgb, mask)  # crops list, list of ndarray(h_i, w_i, 3), dtype=uint8
    return multiscale_cropped_images


def pad_into_square(image):
    width, height = image.size
    new_size = max(width, height)
    new_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))
    left = (new_size - width) // 2
    top = (new_size - height) // 2
    new_image.paste(image, (left, top))
    return new_image


# @brief: save each mask as an individual image
def save_mask_images(mask_image, output_dir, frame_id=None):
    os.makedirs(output_dir, exist_ok=True)
    seg_image_reshape = mask_image.reshape(-1)  # Tensor(H * W)
    ids = torch.unique(seg_image_reshape)  # mask_ids of all detected masks, Tensor(m, ), dtype=uint8
    ids.sort()
    mask_image = mask_image.long()

    for mask_id in ids:
        mask_id = mask_id.item()
        if mask_id == 0:
            continue

        this_mask_image = torch.where(mask_image == mask_id, torch.ones_like(mask_image), torch.zeros_like(mask_image)).float()
        if frame_id is None:
            this_mask_path = os.path.join(output_dir, "%d.png" % mask_id)
        else:
            this_mask_path = os.path.join(output_dir, "%d_%d.png" % (frame_id, mask_id) )
        this_mask_image = this_mask_image[..., None].tile((1, 1, 3))
        this_mask_image = (this_mask_image.cpu().numpy() * 255).astype("uint8")
        cv2.imwrite(this_mask_path, this_mask_image)
