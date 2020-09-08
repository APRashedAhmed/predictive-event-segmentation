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
DIR_LOGS_TB = DIR_LOGS / 'tensorboard'

# Path to the symbolically linked notebook directory
DIR_WBS_SL = DIR_REPO / 'workbooks'

# Path to the symbolically linked notes directory
DIR_NOTES_SL = DIR_REPO / 'notes'

# Path to the models and checkpoint direcotories
DIR_MODELS_SL = DIR_REPO / 'models'

# Path to various locations within the documentation
DIR_DOCS = DIR_REPO / 'docs'
DIR_DOCS_SRC = DIR_DOCS / 'source'
DIR_WBS = DIR_DOCS_SRC / 'workbooks'
DIR_IMGS = DIR_DOCS_SRC / 'images'
DIR_NOTES = DIR_DOCS_SRC / 'notes'

# Path to the models and checkpoint direcotories
DIR_WEIGHTS = DIR_MODELS_SL / 'weights'
DIR_CHECKPOINTS = DIR_MODELS_SL / 'checkpoints'

# To delineate data-specific indices from defaults
_data_prefixes = []

# SCHAPIRO
_data_prefixes.append('_SCH') # Add the schapiro prefix

# Top level Schapiro data dir
DIR_SCH = DIR_DATA_EXT / 'schapiro'
DIR_SCH_FRACTALS = DIR_SCH / 'abstract_discs'
DIR_SCH_FRACTALS_EMB = DIR_SCH / 'abstract_discs_embedded' # resnet101
DIR_SCH_FRACTALS_RS = DIR_SCH / 'abstract_discs_resized_128_160'

# BREAKFAST

# Breakfast directories
_data_prefixes.append('_BK') # Add the breakfast prefix

# Top level breakfast directory
DIR_BK = DIR_DATA_EXT / 'breakfast'
DIR_BK_RAW = DIR_DATA_RAW / 'breakfast'
DIR_BK_META = DIR_BK / 'meta'

# External data
DIR_BK_DATA = DIR_BK / 'Breakfast_data/s1'
DIR_BK_I3D_FVS = DIR_BK / 'i3d_fvs'
DIR_BK_VIDEOS = DIR_BK / 'BreakfastII_15fps_qvga_sync/'

# Raw (processed) data
DIR_BK_64_FRAME_CLIPS = DIR_BK_RAW / '64_frame_clips'

# 64 FV
DIR_BK_64_FV_CLIPS = DIR_BK_64_FRAME_CLIPS / '64dim_fv'
PATH_BK_64_FVS_EVENT_PATHS = DIR_BK_64_FV_CLIPS / 'path_event_data_seed_117.pkl'
PATH_BK_64_FVS_EVENT_DATA = DIR_BK_64_FV_CLIPS / 'event_clips_seed_117.npy'
PATH_BK_64_FVS_NONEVENT_PATHS = DIR_BK_64_FV_CLIPS / 'path_nonevent_data_seed_117.pkl'
PATH_BK_64_FVS_NONEVENT_DATA = DIR_BK_64_FV_CLIPS / 'nonevent_clips_seed_117.npy'

# I3D
DIR_BK_I3D_FV_CLIPS = DIR_BK_64_FRAME_CLIPS / 'i3d_fv'
PATH_BK_I3D_FVS_EVENT_PATHS = DIR_BK_I3D_FV_CLIPS / 'path_event_data_seed_117.pkl'
PATH_BK_I3D_FVS_EVENT_DATA = DIR_BK_I3D_FV_CLIPS / 'event_clips_seed_117.npy'
PATH_BK_I3D_FVS_NONEVENT_PATHS = DIR_BK_I3D_FV_CLIPS / 'path_nonevent_data_seed_117.pkl'
PATH_BK_I3D_FVS_NONEVENT_DATA = DIR_BK_I3D_FV_CLIPS / 'nonevent_clips_seed_117.npy'

# Segmentations
DIR_BK_SEG_COARSE = DIR_BK / 'segmentation_coarse/'
DIR_BK_SEG_FINE = DIR_BK / 'segmentation_fine/'

