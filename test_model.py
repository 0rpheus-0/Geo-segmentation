import torch
from constants import DEVICE, X_TEST_DIR, Y_TEST_DIR
from Dataset import Dataset
import augmentation
from torch.utils.data import DataLoader
from segmentation_models_pytorch import utils


model = torch.jit.load("models_unet_rgb/best_model_new.pt", map_location=DEVICE)


def test_model(model, x_test_dir, y_test_dir):
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        preprocessing=augmentation.preprocessing(augmentation.preprocessing_fn),
    )

    test_dataloader = DataLoader(test_dataset)
    loss = utils.losses.DiceLoss()
    metrics = [utils.metrics.Fscore(), utils.metrics.IoU()]

    test_epoch = utils.train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)
    return logs


if __name__ == "__main__":
    try:
        test_logs = test_model(model, X_TEST_DIR, Y_TEST_DIR)
        print("Test Loss:", test_logs["dice_loss"])
        print("Test F-score:", test_logs["fscore"])
        print("Test IoU:", test_logs["iou_score"])
    except Exception as e:
        print("An error occurred during testing:")
        print(e)
