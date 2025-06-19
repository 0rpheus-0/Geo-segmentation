import numpy as np
import cv2
from glob import glob
from torch.utils.data import Dataset as BaseDataset
import rasterio
from constants import LABEL_COLORS_FILE, CLASSES


class Dataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.images_paths = sorted(glob(f"{images_dir}/*"))
        self.masks_paths = sorted(glob(f"{masks_dir}/*"))

        self.cls_colors = self._get_classes_colors(LABEL_COLORS_FILE)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def _get_classes_colors(self, label_colors_dir):
        cls_colors = {}
        with open(label_colors_dir) as file:
            while line := file.readline():
                R, G, B, label = line.rstrip().split()
                cls_colors[label] = np.array([B, G, R], dtype=np.uint8)

        keyorder = CLASSES
        cls_colors_ordered = {}
        for k in keyorder:
            if k in cls_colors:
                cls_colors_ordered[k] = cls_colors[k]
            else:
                raise ValueError(f"unexpected label {k}, cls colors: {cls_colors}")

        return cls_colors_ordered

    def __getitem__(self, i):
        imgdata = rasterio.open(self.images_paths[i])
        image = np.array(imgdata.read())
        image = image.transpose(1, 2, 0)

        mask = cv2.imread(self.masks_paths[i])
        masks = [cv2.inRange(mask, color, color) for color in self.cls_colors.values()]
        masks = [(m > 0).astype("float32") for m in masks]
        mask = np.stack(masks, axis=-1).astype("float")

        image = np.pad(image, ((1, 1), (0, 0), (0, 0)), mode="edge")
        mask = np.pad(mask, ((1, 1), (0, 0), (0, 0)), mode="edge")

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.images_paths)
