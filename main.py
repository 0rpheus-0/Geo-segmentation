import numpy as np
import torch
import random
from segmentation_models_pytorch import utils
from constants import X_TRAIN_DIR, Y_TRAIN_DIR
from Dataset import Dataset
import visual

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

loss = utils.losses.DiceLoss()
      
dataset = Dataset(X_TRAIN_DIR, Y_TRAIN_DIR)
image, mask = dataset[np.random.randint(len(dataset))]
visual.visualize_multichennel_mask(image, mask)