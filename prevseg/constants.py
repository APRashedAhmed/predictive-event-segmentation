"""Place to define constants for the dataset"""
from argparse import Namespace

from prevseg import index

DEFAULT_SEED = 117

# Breakfast video info
DIM_H = 240
DIM_W = 320
DIM_C = 3

# Experiment  Defaults
DEFAULT_MODEL_NAME = 'prednet'
DEFAULT_N_LAYERS = 4
DEFAULT_INPUT_SIZE = 2048
DEFAULT_TIME_STEPS = 64
DEFAULT_LR = 0.000333
DEFAULT_OUTPUT_MODE = 'error'
DEFAULT_DEVICE = 'cuda'
DEFAULT_N_TEST = 16
DEFAULT_N_VAL = 256
DEFAULT_BATCH_SIZE = 256
DEFAULT_N_EPOCHS = 10
DEFAULT_N_WORKERS = 4
DEFAULT_LAYER_LOSS_MODE = 'first'

DEFAULT_BK_DATALOADER = 'BreakfastI3DFVDataset'

DEFAULT_HPARAMS = Namespace(**{
    'model_name' : DEFAULT_MODEL_NAME,
    'n_layers' : DEFAULT_N_LAYERS,
    'input_size' : DEFAULT_INPUT_SIZE,
    'time_steps' : DEFAULT_TIME_STEPS,
    'dir_checkpoints' : str(index.DIR_CHECKPOINTS),
    'dir_weights' : str(index.DIR_WEIGHTS),
    'dir_logs' : str(index.DIR_LOGS_TB),
    'lr' : DEFAULT_LR,
    'output_mode' : DEFAULT_OUTPUT_MODE,
    'device' : DEFAULT_DEVICE,
    'n_test' : DEFAULT_N_TEST,
    'n_val' : DEFAULT_N_VAL,
    'seed' : DEFAULT_SEED,
    'batch_size' : DEFAULT_BATCH_SIZE,
    'n_epochs' : DEFAULT_N_EPOCHS,
    'n_workers' : DEFAULT_N_WORKERS,
    'layer_loss_mode' : DEFAULT_LAYER_LOSS_MODE,
})


