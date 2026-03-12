import torch
from constants import DEVICE, X_TEST_DIR, Y_TEST_DIR
from Dataset import Dataset
import augmentation
from torch.utils.data import DataLoader
from segmentation_models_pytorch import utils
import warnings
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


model = torch.jit.load("models_unet_rgb/best_model_new.pt", map_location=DEVICE)


def test_model(model, x_test_dir, y_test_dir, progress_callback=None):
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        preprocessing=augmentation.preprocessing(augmentation.preprocessing_fn),
    )

    test_dataloader = DataLoader(test_dataset)
    loss_fn = utils.losses.DiceLoss()
    metrics = [utils.metrics.Fscore(), utils.metrics.IoU()]
    model.eval()
    n = len(test_dataloader)
    loss_sum = 0
    fscore_sum = 0
    iou_sum = 0

    for i, (x, y) in enumerate(test_dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        with torch.no_grad():
            pr = model(x)
            loss = loss_fn(pr, y)
            fscore = metrics[0](pr, y)
            iou = metrics[1](pr, y)
        loss_sum += loss.item()
        fscore_sum += fscore.item()
        iou_sum += iou.item()
        progress = (i + 1) / n
        if progress_callback:
            progress_callback(progress)
        else:
            bar_len = 30
            filled_len = int(bar_len * progress)
            bar = "█" * filled_len + "-" * (bar_len - filled_len)
            print(f"\rТестирование: |{bar}| {int(progress * 100)}%", end="")
    if not progress_callback:
        print()
    logs = {
        "dice_loss": loss_sum / n,
        "fscore": fscore_sum / n,
        "iou_score": iou_sum / n,
    }
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
