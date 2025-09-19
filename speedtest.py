import torch
import numpy as np
import rasterio
from rasterio.plot import adjust_band
from constants import DEVICE, CLASSES, INFER_WIDTH, INFER_HEIGHT, INFER_CHANNEL
import sys
import os
import time

colors_imshow = {
    "City": np.array([170, 240, 209]),
    "Cloud": np.array([184, 61, 245]),
    "Field": np.array([242, 242, 13]),
    "Sand": np.array([241, 184, 22]),
    "Trees": np.array([63, 220, 32]),
    "Water": np.array([13, 160, 236]),
}


def color_mask(mask: np.ndarray):
    mask = mask.squeeze()
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_code, cls in enumerate(CLASSES):
        cls_mask = mask == cls_code
        colored_mask += np.multiply.outer(cls_mask, colors_imshow[cls]).astype(np.uint8)

    return colored_mask


unet = torch.jit.load("models_unet_rgb/best_model_new.pt", map_location=DEVICE)

try:
    data_path = sys.argv[1]
    result_path = sys.argv[2]
except IndexError:
    print("Input <data path> <result path>")

images_path = os.listdir(data_path)

execution_proc = 0
execution_pred = 0
start = time.time()
for image_path in images_path:
    with rasterio.open(data_path + image_path) as image_data:
        image = np.array(adjust_band(image_data.read()))
        image = image.astype("float32")

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask_unet = unet(x_tensor)
    pr_mask_unet = pr_mask_unet.squeeze().cpu().detach().numpy()
    pr_mask_unet = np.argmax(pr_mask_unet, axis=0)

    mask = color_mask(pr_mask_unet)
    mask = mask.transpose(2, 0, 1)

    meta = {
        "width": INFER_WIDTH,
        "height": INFER_HEIGHT,
        "count": INFER_CHANNEL,
        "dtype": "uint8",
    }
    result_name = result_path + "mask_" + image_path
    with rasterio.open(result_name, "w", **meta) as mask_data:
        mask_data.write(mask)
end = time.time()
execution = end - start
print(f"Time: {execution} sec\nSpeed: {len(images_path) / execution} image/sec")
