import albumentations as albu


preprocessing_fn = lambda img, **kwargs: img.astype("float32") / 255


def training_ablumentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.OneOf(
            [
                albu.Sharpen(alpha=(0.1, 0.2), lightness=(0.1, 0.2), p=0.5),
                albu.Blur(blur_limit=[3, 5], p=0.5),
            ],
            p=0.7,
        ),
        albu.RandomBrightnessContrast(
            brightness_limit=0.05, contrast_limit=0.05, p=0.5
        ),
    ]
    return albu.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
