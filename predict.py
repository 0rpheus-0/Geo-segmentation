import torch
import numpy as np
import rasterio
from rasterio.plot import adjust_band
from constants import DEVICE
import sys
import visual


images_paths = sys.argv[1]
unet = torch.jit.load("models_unet/best_model_new.pt", map_location=DEVICE)

imgdata = rasterio.open(images_paths)
image = np.array([adjust_band(imgdata.read(1))])
image = image.astype("float32")

x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
pr_mask_unet = unet(x_tensor)
pr_mask_unet = pr_mask_unet.squeeze().cpu().detach().numpy()


visual.visualize_compere_predict(image, np.argmax(pr_mask_unet, axis=0))
