import numpy as np
import matplotlib.pyplot as plt
from constants import CLASSES

colors_imshow = {
    "City": np.array([170, 240, 209]),
    "Cloud": np.array([184, 61, 245]),
    "Field": np.array([242, 242, 13]),
    "Sand": np.array([241, 184, 22]),
    "Trees": np.array([63, 220, 32]),
    "Water": np.array([13, 160, 236]),
}


def visual_mask(mask: np.ndarray):
    sc_mask = np.zeros((mask[0].shape[0], mask[0].shape[1], 3), dtype=np.uint8)
    square_ratios = {}
    for i, singlechannel_mask in enumerate(mask):
        cls = CLASSES[i]
        singlechannel_mask = singlechannel_mask.squeeze()
        square_ratios[cls] = singlechannel_mask.sum() / singlechannel_mask.size
        sc_mask += np.multiply.outer(singlechannel_mask > 0, colors_imshow[cls]).astype(
            np.uint8
        )

    title = "Square: " + "\n".join(
        [f"{cls}: {square_ratios[cls] * 100:.1f}%" for cls in CLASSES]
    )
    return sc_mask, title


def visualize_multichennel_mask(image: np.ndarray, multichennel_mask: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image[0], cmap="grey")
    mask_to_show, title = visual_mask(multichennel_mask)
    axes[1].imshow(mask_to_show)
    axes[1].set_title(title)
    plt.tight_layout()
    plt.show()


def color_mask(mask: np.ndarray):
    mask = mask.squeeze()
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    square_ratios = {}
    for cls_code, cls in enumerate(CLASSES):
        cls_mask = mask == cls_code
        square_ratios[cls] = cls_mask.sum() / cls_mask.size
        colored_mask += np.multiply.outer(cls_mask, colors_imshow[cls]).astype(np.uint8)

    return colored_mask, square_ratios


def reverse_normalize(img, mean, std):
    img = img * np.array(std) + np.array(mean)
    return img


def visualize_result(img: np.ndarray, mask_gt: np.ndarray, mask_pred: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(img[0], cmap="grey")

    mask_gt, square_ratios = color_mask(mask_gt)
    title = "Square:\n" + "\n".join(
        [f"{cls}: {square_ratios[cls] * 100:.1f}%" for cls in CLASSES]
    )
    axes[1].imshow(mask_gt, cmap="twilight")
    axes[1].set_title(title)

    mask_pred, square_ratios = color_mask(mask_pred)
    title = "Square:\n" + "\n".join(
        [f"{cls}: {square_ratios[cls] * 100:.1f}%" for cls in CLASSES]
    )
    axes[2].imshow(mask_pred, cmap="twilight")
    axes[2].set_title(title)

    plt.tight_layout()
    plt.show()


def visualize_predict(img: np.ndarray, mask_pred: np.ndarray):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img[0], cmap="grey")

    mask_pred, square_ratios = color_mask(mask_pred)
    title = "Square:\n" + "\n".join(
        [f"{cls}: {square_ratios[cls] * 100:.1f}%" for cls in CLASSES]
    )
    axes[1].imshow(mask_pred, cmap="twilight")
    axes[1].set_title(title)
    axes[1].legend(handles=[])
    plt.tight_layout()
    plt.show()


def visualize_compere_predict(
    img: np.ndarray, mask_pred_1: np.ndarray, mask_pred_2: np.ndarray
):
    _, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(img[0], cmap="grey")
    axes[1].imshow(img[0], cmap="grey")
    axes[2].imshow(img[0], cmap="grey")

    mask_pred_1, _ = color_mask(mask_pred_1)
    axes[1].imshow(mask_pred_1, cmap="twilight", alpha=0.5)
    axes[1].set_title("Unet")
    axes[1].legend(handles=[])

    mask_pred_2, _ = color_mask(mask_pred_2)
    axes[2].imshow(mask_pred_2, cmap="twilight", alpha=0.5)
    axes[2].set_title("FNP")
    axes[2].legend(handles=[])

    plt.tight_layout()
    plt.show()
