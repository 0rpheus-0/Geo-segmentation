import torch
from constants import DEVICE, X_TEST_DIR, Y_TEST_DIR, X_TRAIN_DIR, Y_TRAIN_DIR
from Dataset import Dataset
import augmentation
from torch.utils.data import DataLoader
from segmentation_models_pytorch import utils
import numpy as np
import visual

best_model = torch.jit.load("models_unet/best_model_new.pt", map_location=DEVICE)

test_dataset = Dataset(
    X_TRAIN_DIR,
    Y_TRAIN_DIR,
    preprocessing=augmentation.preprocessing(augmentation.preprocessing_fn),
)

test_dataloader = DataLoader(test_dataset)
loss = utils.losses.DiceLoss()
metrics = [utils.metrics.Fscore(), utils.metrics.IoU()]

test_epoch = utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)
