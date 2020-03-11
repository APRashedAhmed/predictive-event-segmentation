"""Index script with paths to different parts of ssdl"""
from pathlib import Path

# Define some Path objects to folders within the repo
# Path to the repo as a whole. Implicitly assume this file is two directories
# into the full repo.
DIR_REPO = Path(__file__).resolve().parent.parent

# Path to the top-level data directories which contain the data
DIR_DATA = DIR_REPO / 'data'
DIR_DATA_EXT = DIR_DATA / 'external'
DIR_DATA_INT = DIR_DATA / 'interim'
DIR_DATA_PROC = DIR_DATA / 'processed'
DIR_DATA_RAW = DIR_DATA / 'raw'

# Path to the log file directory
DIR_LOGS = DIR_REPO / 'logs'

# Path to the symbolically linked notebook and report directories
DIR_WBS_SL = DIR_REPO / 'workbooks'

# Path to the models and checkpoint direcotories
DIR_MODELS_SL = DIR_REPO / 'models'

# Path to various locations within the documentation
DIR_DOCS = DIR_REPO / 'docs'
DIR_DOCS_SRC = DIR_DOCS / 'source'
DIR_WBS = DIR_DOCS_SRC / 'notebooks'
DIR_IMGS = DIR_DOCS_SRC / 'images'

# Top level breakfast directory
DIR_BREAKFAST = DIR_DATA_EXT / 'breakfast'
DIR_RAW_BREAKFAST = DIR_DATA_RAW / 'breakfast'
DIR_BREAKFAST_META = DIR_BREAKFAST / 'meta'

# External data
DIR_BREAKFAST_DATA = DIR_BREAKFAST / 'Breakfast_data/s1'
DIR_I3D_FVS = DIR_BREAKFAST / 'i3d_fvs'
DIR_BREAKFAST_VIDEOS = DIR_BREAKFAST / 'BreakfastII_15fps_qvga_sync/'

# Raw (processed) data
DIR_64_FRAME_CLIPS = DIR_RAW_BREAKFAST / '64_frame_clips'

DIR_64_FV_CLIPS = DIR_64_FRAME_CLIPS / '64dim_fv'
PATH_64_FVS_EVENT_PATHS = DIR_64_FV_CLIPS / 'path_event_data_seed_117.pkl'
PATH_64_FVS_EVENT_DATA = DIR_64_FV_CLIPS / 'event_clips_seed_117.npy'
PATH_64_FVS_NONEVENT_PATHS = DIR_64_FV_CLIPS / 'path_nonevent_data_seed_117.pkl'
PATH_64_FVS_NONEVENT_DATA = DIR_64_FV_CLIPS / 'nonevent_clips_seed_117.npy'

DIR_I3D_FV_CLIPS = DIR_64_FRAME_CLIPS / 'i3d_fv'
PATH_I3D_FVS_EVENT_PATHS = DIR_I3D_FV_CLIPS / 'path_event_data_seed_117.pkl'
PATH_I3D_FVS_EVENT_DATA = DIR_I3D_FV_CLIPS / 'event_clips_seed_117.npy'
PATH_I3D_FVS_NONEVENT_PATHS = DIR_I3D_FV_CLIPS / 'path_nonevent_data_seed_117.pkl'
PATH_I3D_FVS_NONEVENT_DATA = DIR_I3D_FV_CLIPS / 'nonevent_clips_seed_117.npy'


# Segmentations
DIR_COARSE_SEG = DIR_BREAKFAST / 'segmentation_coarse/'
DIR_FINE_SEG = DIR_BREAKFAST / 'segmentation_fine/'

# Path to the models and checkpoint direcotories
DIR_WEIGHTS = DIR_MODELS_SL / 'weights'
DIR_CHECKPOINTS = DIR_MODELS_SL / 'checkpoints'

# Logs
DIR_LOGS_TB = DIR_LOGS / 'tensorboard'

