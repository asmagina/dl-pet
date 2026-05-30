DATA_ROOT = "/root/data"
# CATEGORY = "bottle"
IMAGE_SIZE = (640, 959)
BATCH_SIZE = 1
NUM_WORKERS = 4

NUM_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

CHECKPOINT_DIR = "runs/checkpoints"
BEST_CKPT = "models/unet_carvana_scale0.5_epoch2.pth"  # "runs/checkpoints/best.pt"


# Test only: use None to evaluate the full test split.
TEST_START_INDEX = None
TEST_STOP_INDEX = None

LOG_DIR = "runs/tensorboard"