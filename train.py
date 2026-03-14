import numpy as np
import torch
import random
from segmentation_models_pytorch import utils
from constants import (
    X_TRAIN_DIR,
    Y_TRAIN_DIR,
    ENCODER,
    ENCODER_WEIGHTS,
    CLASSES,
    ACTIVATION,
    X_VALID_DIR,
    Y_VALID_DIR,
    INIT_LR,
    DEVICE,
    EPOCHS,
    BATCH_SIZE,
    INFER_HEIGHT,
    INFER_WIDTH,
    LR_DECREASE_STEP,
    LR_DECREASE_COEF,
    INFER_CHANNEL,
)
from Dataset import Dataset
import augmentation
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import warnings
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def train_model(
    x_train_dir,
    y_train_dir,
    x_valid_dir,
    y_valid_dir,
    encoder,
    encoder_weights,
    classes,
    activation,
    init_lr,
    epochs,
    batch_size,
    infer_height,
    infer_width,
    lr_decrease_step,
    lr_decrease_coef,
    infer_channel,
    model_save_dir="models_unet_rgb",
    model_save_name="best_model_2",
    plot_path="losses_IOU.png",
    seed=42,
    progress_callback=None,
):
    """
    Обучает модель сегментации и сохраняет лучшую модель.
    Возвращает логи потерь и метрик.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    loss = utils.losses.DiceLoss()
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=len(classes),
        activation=activation,
    )
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=augmentation.training_ablumentation(),
        preprocessing=augmentation.preprocessing(augmentation.preprocessing_fn),
    )
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        preprocessing=augmentation.preprocessing(augmentation.preprocessing_fn),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    metrics = [utils.metrics.Fscore(), utils.metrics.IoU()]
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=init_lr),
        ]
    )
    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )
    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )
    max_score = 0
    loss_logs = {"train": [], "val": []}
    metric_logs = {"train": [], "val": []}

    for i in range(0, epochs):
        print(f"\nEpoch: {i}")
        train_logs = train_epoch.run(train_loader)
        train_loss, train_metric, train_metric_IOU = list(train_logs.values())
        loss_logs["train"].append(train_loss)
        metric_logs["train"].append(train_metric_IOU)

        valid_logs = valid_epoch.run(valid_loader)
        val_loss, val_metric, val_metric_IOU = list(valid_logs.values())
        loss_logs["val"].append(val_loss)
        metric_logs["val"].append(val_metric_IOU)

        if max_score < valid_logs["iou_score"]:
            max_score = valid_logs["iou_score"]
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(model, os.path.join(model_save_dir, model_save_name + ".pth"))
            trace_image = torch.randn(
                batch_size, infer_channel, infer_height, infer_width
            )
            traced_model = torch.jit.trace(model, trace_image.to(DEVICE))
            torch.jit.save(
                traced_model, os.path.join(model_save_dir, model_save_name + ".pth")
            )
            print("Model saved!")

        print("LR:", optimizer.param_groups[0]["lr"])
        if i > 0 and i % lr_decrease_step == 0:
            print("Decrease decoder learning rate")
            optimizer.param_groups[0]["lr"] /= lr_decrease_coef

        # Вызов колбэка для GUI
        if progress_callback is not None:
            progress_callback(
                epoch=i,
                total_epochs=epochs,
                loss_logs=loss_logs,
                metric_logs=metric_logs,
            )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(loss_logs["train"], label="train")
    axes[0].plot(loss_logs["val"], label="val")
    axes[0].set_title("losses - Dice")

    axes[1].plot(metric_logs["train"], label="train")
    axes[1].plot(metric_logs["val"], label="val")
    axes[1].set_title("IOU")

    [ax.legend() for ax in axes]
    fig.savefig(plot_path, dpi=300)
    plt.show()
    return loss_logs, metric_logs


if __name__ == "__main__":
    try:
        train_model(
            x_train_dir=X_TRAIN_DIR,
            y_train_dir=Y_TRAIN_DIR,
            x_valid_dir=X_VALID_DIR,
            y_valid_dir=Y_VALID_DIR,
            encoder=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=CLASSES,
            activation=ACTIVATION,
            init_lr=INIT_LR,
            epochs=2,
            batch_size=BATCH_SIZE,
            infer_height=INFER_HEIGHT,
            infer_width=INFER_WIDTH,
            lr_decrease_step=LR_DECREASE_STEP,
            lr_decrease_coef=LR_DECREASE_COEF,
            infer_channel=INFER_CHANNEL,
        )
    except Exception as e:
        import traceback

        print("An error occurred during training:")
        print(e)
