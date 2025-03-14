import torch
from constants import DEVICE, X_TEST_DIR, Y_TEST_DIR
from Dataset import Dataset
import augmentation
from torch.utils.data import DataLoader
from segmentation_models_pytorch import utils
import numpy as np
import visual

best_model = torch.jit.load("models/best_model_new.pt", map_location=DEVICE)

test_dataset = Dataset(
    X_TEST_DIR,
    Y_TEST_DIR,
    augmentation=augmentation.training_ablumentation(),
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


for i in range(20):
    n = np.random.choice(len(test_dataset))

    image, gt_mask = test_dataset[n]
    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model(x_tensor)
    pr_mask = pr_mask.squeeze().cpu().detach().numpy()

    label_mask = np.argmax(pr_mask, axis=0)

    visual.visualize_result(image, np.argmax(gt_mask, axis=0), label_mask)
