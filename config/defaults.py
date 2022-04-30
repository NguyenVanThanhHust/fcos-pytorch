from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.PRIOR = 0.01
_C.GAMMA = 2.0
_C.ALPHA = 0.25
_C.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.SIZES = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]]
_C.POS_RADIUS = 1.5
_C.CENTER_SAMPLING = True

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 80
_C.MODEL.BACKBONE="ResNet50FPN"
_C.MODEL.STRIDE = 128
_C.MODEL.NUM_CONV_BLOCK = 4
_C.MODEL.FEATURE_DEPTH = 256

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.FOLDER = "../coco/"
# Max size of image
_C.INPUT.MAX_SIZE = 1333
# Size of the image during training
_C.INPUT.SIZE_TRAIN = 512
# Size of the image during test
_C.INPUT.SIZE_TEST = 512
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = (0.485, 0.456, 0.406)
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 2

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_EPOCHS = 2
_C.SOLVER.WARMUP_TYPE = "CosineAnnealingWarmRestarts"

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""