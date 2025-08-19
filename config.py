import torch

# --- Reproducibility ---
# setting a seed ensures that random operations (like model weight
# initialisation, data shuffling) are the same every time we run the code
SEED = 42

# -- Project Paths --
DATA_ROOT = '/home/qzhang-server2/Documents/Abdullah_Data/Frame_Dataset'
OUTPUT_DIR = '/home/qzhang-server2/Documents/Abdullah_Notebooks/sb_baseline/results'

# --- Data & Augmentation Parameters ---
RESOLUTION = 480

# -- Model & Training Parameters --
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 2  # 1 class (side-branch) + 1 background
BATCH_SIZE = 2
NUM_EPOCHS = 40  
LEARNING_RATE = 0.005 # relying on LR scheduler to decrease this over time
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# -- Dataloader Parameters --
NUM_WORKERS = 6 

# --- Inference Parameters ---
# centralising the confidence threshold for visualisation/evaluation
# is good practice.
CONFIDENCE_THRESHOLD = 0.5 # starting at 0.5 and increasing if needed

# --- Temporal Model Parameters ---
# number of adjacent frames to use on EACH side of the central frame
# k=2 means we use 5 frames total: (n-2, n-1, n, n+1, n+2)
TEMPORAL_FRAMES_K = 3

# CHANGE 'PATIENCE' (early stoppage) value in both utils.py and train.py