import torch
from constants import DEVICE, X_TEST_DIR, Y_TEST_DIR
from Dataset import Dataset
import augmentation
from torch.utils.data import DataLoader
from segmentation_models_pytorch import utils


best_model = torch.jit.load("models_unet_rgb/best_model_new.pt", map_location=DEVICE)

test_dataset = Dataset(
    X_TEST_DIR,
    Y_TEST_DIR,
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
