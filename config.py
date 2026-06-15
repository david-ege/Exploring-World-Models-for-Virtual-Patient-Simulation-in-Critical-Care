# Configs for data, moedels, training and locations
import data.constants as constants

#Const
CARDIOVASCULAR_DATA_PATH = '/home/bbe9928/thesis_work/hirid_jepa/data/Cardiovascular_common_subset.h5'
DATASET_DATA_PATH = '/home/bbe9928/thesis_work/hirid_jepa/data/common_stage_scaled.h5'
CHECKPOINT_DIR = '/home/bbe9928/thesis_work/hirid_jepa/checkpoints'
RESULTS_DIR = '/home/bbe9928/thesis_work/hirid_jepa/results'

DATA_PATH = DATASET_DATA_PATH

#Checkpoints
BEST_CHECKPOINT = 'gru_ctx36_tgt36_h256_l1_08_06_15-44.pt' #model checkpoint used for mortality classifier
BEST_CLASSIFIER_CHECKPOINT = 'gru_classifier_h64_do0.2_08_06_12-13.pt' #used by world model / rl / cem
def get_checkpoint_path(checkpoint = BEST_CHECKPOINT):
    return f"{CHECKPOINT_DIR}/{checkpoint}"

# Shared Settings -------------------------------------------------------------------------------
# Data
CONTEXT_STEPS = 36      # 3 hours of context
TARGET_STEPS = 36       # 1 hour prediction

# Variable subset — set to None to use all variables
MEASUREMENT_SUBSET = None   # e.g. INFORMATIVE_MEASUREMENT_IDX
TREATMENT_SUBSET   = None   # e.g. INFORMATIVE_TREATMENT_IDX

#Predictor Settings -------------------------------------------------------------------------------
# Model
PRED_ENCODER_DIM = 64
PRED_HIDDEN_DIM = 256
PRED_NUM_LAYERS = 1
PRED_DROPOUT = 0.3

# Input flags
PRED_USE_CONTEXT_MASK = False  # concatenate binary observation mask to GRU input
PRED_USE_DELTA_T      = False  # concatenate scaled time-since-last-observation to GRU input

# Training
PRED_LEARNING_RATE = 1e-4
PRED_WEIGHT_DECAY = 1e-4
PRED_BATCH_SIZE = 64
PRED_NUM_EPOCHS = 30
PRED_GRAD_CLIP = 1.0
PRED_PATIENCE = 7

# --- Classifier (GRU & LR)---
CLASSIFIER_C = 1.0 # C for LR
# Classifier model
CLASSIFIER_HIDDEN_DIM     = 64
CLASSIFIER_NUM_LAYERS     = 1
CLASSIFIER_DROPOUT        = 0.2
CLASSIFIER_USE_CONTEXT_MASK = False
CLASSIFIER_USE_DELTA_T      = False
#Training
CLASSIFIER_LEARNING_RATE = 3e-4
CLASSIFIER_WEIGHT_DECAY  = 1e-4
CLASSIFIER_BATCH_SIZE    = 64
CLASSIFIER_NUM_EPOCHS    = 1000
CLASSIFIER_GRAD_CLIP     = 1.0
CLASSIFIER_PATIENCE      = 12

#CEM
CEM_START_STEP = 0 #in 5min frames
CEM_NUM_STEPS = 3 #3*36 5min steps =9h in die Zukunft (+3h timeframe zum starten)
CEM_DATASET = CARDIOVASCULAR_DATA_PATH

CEM_BATCH_SIZE = 300
CEM_NUM_ITER = 30
CEM_ELITE_FRAC = .1

CEM_INIT_STDEV = 1.0
CEM_EXTRA_STDEV  = 0.5
CEM_STDEV_DECAY_TIME = 25

# Device
DEVICE = 'cuda'