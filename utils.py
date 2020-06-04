# -*- coding: utf-8 -*-
import numpy as np

from PIL import Image
from config import cfg

rgb_mean = cfg.RGB_MEAN
rgb_std  = cfg.RGB_STD
colormap = cfg.COLORMAP
classes  = cfg.CLASSES

def visual_result(image, label, alpha=0.7):
    """
    image shape -> [H, W, C]
    label shape -> [H, W]
    """
    image = (image * rgb_std + rgb_mean) * 255
    image, label = image.astype(np.int), label.astype(np.int)
    H, W, C = image.shape
    masks_color = np.zeros(shape=[H, W, C])
    inv_masks_color = np.zeros(shape=[H, W, C])
    cls = []
    for i in range(H):
        for j in range(W):
            cls_idx = label[i, j]
            masks_color[i, j] = np.array(colormap[cls_idx])
            cls.append(cls_idx)
            if classes[cls_idx] == "background":
                inv_masks_color[i, j] = alpha * image[i, j]

    show_image = np.zeros(shape=[224, 672, 3])
    cls = set(cls)
    for x in cls:
        print("=> ", classes[x])
    show_image[:, :224, :] = image
    show_image[:, 224:448, :] = masks_color
    show_image[:, 448:, :] = (1-alpha)*image + alpha*masks_color + inv_masks_color
    show_image = Image.fromarray(np.uint8(show_image))
    return show_image