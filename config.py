# Configs for data, moedels, training and locations
import data.constants as constants

DATA_PATH = '/home/bbe9928/thesis_work/hirid_jepa/data/Surgical_Neurological_subset.h5'
CHECKPOINT_DIR = '/home/bbe9928/thesis_work/hirid_jepa/checkpoints'

# Data
CONTEXT_STEPS = 36      # 3 hours of context
TARGET_STEPS = 36       # 1 hour prediction

# Variable subset — set to None to use all variables
MEASUREMENT_SUBSET = constants.TOP_14_STATISTICAL_RELEVANCE_MEASUREMENT_IDX   # e.g. INFORMATIVE_MEASUREMENT_IDX
TREATMENT_SUBSET   = constants.TOP_14_STATISTICAL_RELEVANCE_TREATMENT_IDX   # e.g. INFORMATIVE_TREATMENT_IDX

# Model
ENCODER_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.3


# Training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 30
GRAD_CLIP = 1.0
PATIENCE = 7

# Device
DEVICE = 'cuda'