import torch

DATSET_NAME = "geo_dataset"

X_TRAIN_DIR = f"{DATSET_NAME}/train"
Y_TRAIN_DIR = f"{DATSET_NAME}/train_mask/defaultannot"

X_VALID_DIR = f"{DATSET_NAME}/validation"
Y_VALID_DIR = f"{DATSET_NAME}/validation_mask/defaultannot"

X_TEST_DIR = f"{DATSET_NAME}/test"
Y_TEST_DIR = f"{DATSET_NAME}/test_mask/defaultannot"

LABEL_COLORS_FILE = f"{DATSET_NAME}/label_colors.txt"

CLASSES = ["City", "Cloud", "Field", "Sand", "Trees", "Water"]
ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "softmax2d"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 32

INIT_LR = 0.0005
LR_DECREASE_STEP = 15
LR_DECREASE_COEF = 2

INFER_WIDTH = 384
INFER_HEIGHT = 384
