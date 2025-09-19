import torch
import numpy as np
import rasterio
from rasterio.plot import adjust_band
from constants import DEVICE, INFER_HEIGHT, INFER_WIDTH
import sys
import visual
import cv2
import matplotlib.pyplot as plt


def pad_image(image, target_size):
    _, H, W = image.shape
    tile_h, tile_w = target_size

    pad_h = (tile_h - H % tile_h) % tile_h
    pad_w = (tile_w - W % tile_w) % tile_w

    padded_image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric")

    return padded_image


def split_image(image, tile_size):
    _, H, W = image.shape
    tile_h, tile_w = tile_size

    tiles = []
    positions = []

    for y in range(0, H, tile_h):
        for x in range(0, W, tile_w):
            tile = image[:, y : y + tile_h, x : x + tile_w]
            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions


def merge_mask(masks, positions, full_shape, tile_size):
    C = masks[0].shape[0]
    _, H, W = full_shape
    tile_h, tile_w = tile_size

    merged = np.zeros((C, H, W), dtype=np.float32)

    for tile, (y, x) in zip(masks, positions):
        merged[:, y : y + tile_h, x : x + tile_w] = tile

    return merged


def crop_mask(mask, original_shape):
    _, H, W = original_shape
    return mask[:, :H, :W]


images_paths = sys.argv[1]
unet = torch.jit.load("models_unet_rgb/best_model_new.pt", map_location=DEVICE)

# todo проверка на типы

image_f = rasterio.open(images_paths)
image = np.array(adjust_band(image_f.read([1, 2, 3])))
image = adjust_band(np.exp(2 * image))
image = image.astype("float32")

padding_image = pad_image(image, (INFER_HEIGHT, INFER_WIDTH))
tiles, positions = split_image(padding_image, (INFER_HEIGHT, INFER_WIDTH))

masks = []
for tile in tiles:
    x_tensor = torch.from_numpy(tile).to(DEVICE).unsqueeze(0)
    pr_mask_unet = unet(x_tensor)
    pr_mask_unet = pr_mask_unet.squeeze().cpu().detach().numpy()
    masks.append(pr_mask_unet)

mask = merge_mask(masks, positions, padding_image.shape, (INFER_HEIGHT, INFER_WIDTH))
mask = crop_mask(mask, image.shape)

visual.visualize_compere_predict(image.transpose(1, 2, 0), np.argmax(mask, axis=0))

plt.show()
