import numpy as np
import torch
import random
from segmentation_models_pytorch import utils
from constants import X_TRAIN_DIR, Y_TRAIN_DIR, ENCODER, ENCODER_WEIGHTS, CLASSES, ACTIVATION, X_VALID_DIR, Y_VALID_DIR, INIT_LR, DEVICE, EPOCHS, BATCH_SIZE,  INFER_HEIGHT, INFER_WIDTH, LR_DECREASE_STEP, LR_DECREASE_COEF
from Dataset import Dataset
import visual
import augmentation
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import warnings
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

loss = utils.losses.DiceLoss()
      
# dataset = Dataset(X_TRAIN_DIR, Y_TRAIN_DIR)

# for i in range(5):
#     image, mask = dataset[np.random.randint(len(dataset))]
#     visual.visualize_multichennel_mask(image, mask)

# augmented_dataset = Dataset(
#     X_TRAIN_DIR, 
#     Y_TRAIN_DIR, 
#     augmentation=augmentation.training_augmentation()
# )

# indx = np.random.randint(len(augmented_dataset))

# for i in range(3):
#     image, mask = augmented_dataset[indx]
#     visual.visualize_multichennel_mask(image, mask)

model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
    in_channels=1,
)    

# print(model)

preprocessing_fn = lambda img, **kwargs: img.astype("float32") / 255

# print(preprocessing_fn)

train_dataset = Dataset(
    X_TRAIN_DIR, 
    Y_TRAIN_DIR, 
    augmentation=augmentation.training_augmentation(), 
    preprocessing=augmentation.get_preprocessing(preprocessing_fn)
)

valid_dataset = Dataset(
    X_VALID_DIR, 
    Y_VALID_DIR,  
     preprocessing=augmentation.get_preprocessing(preprocessing_fn)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) 
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
metrics = [
    utils.metrics.Fscore(),
    utils.metrics.IoU()
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=INIT_LR),
])

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
for i in range(0, EPOCHS):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    train_loss, train_metric, train_metric_IOU = list(train_logs.values())
    loss_logs["train"].append(train_loss)
    metric_logs["train"].append(train_metric_IOU)

    valid_logs = valid_epoch.run(valid_loader)
    val_loss, val_metric, val_metric_IOU = list(valid_logs.values())
    loss_logs["val"].append(val_loss)
    metric_logs["val"].append(val_metric_IOU)
    
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, 'models/best_model_new.pth')
        trace_image = torch.randn(BATCH_SIZE, 3, INFER_HEIGHT, INFER_WIDTH)
        traced_model = torch.jit.trace(model, trace_image.to(DEVICE))
        torch.jit.save(traced_model, 'models/best_model_new.pt')
        print('Model saved!')

    print("LR:", optimizer.param_groups[0]['lr'])
    if i > 0 and i % LR_DECREASE_STEP == 0:
        print('Decrease decoder learning rate')
        optimizer.param_groups[0]['lr'] /= LR_DECREASE_COEF