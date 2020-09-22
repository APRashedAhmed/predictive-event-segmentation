"""Script for the breakfast dataloader"""
import logging
from pathlib import Path

# import cv2
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.sampler import SubsetRandomSampler

from prevseg import index
from prevseg.utils import isiterable
from prevseg.constants import DIM_H, DIM_W, DIM_C, DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class _BreakfastClipsDataset(Dataset):
    def __init__(self, data_names_path, data_path, half=False, first=True):
        self.half = half
        self.first = first
        self.half_len = self.fv_len // 2
        # Data names path
        if not isiterable(data_names_path):
            data_names_path = [data_names_path]
            
        self.data_names_path = [Path(path) for path in data_names_path]
        
        paths_list = []
        for path in self.data_names_path:
            with open(str(path), 'rb') as file:
                paths_list.append(pickle.load(file))

        self.paths = np.concatenate(paths_list)

        # Data Paths
        if not isiterable(data_path):
            data_path = [data_path]
            
        self.data_path = [Path(path) for path in data_path]
        self.data = np.concatenate(
            [np.load(path) for path in self.data_path])

        # Set getitem depending on we want the data
        if self.half and self.first:
            self._getitem = lambda idx : (self.data[idx, :, self.half_len:],
                                          str(self.paths[idx]))
        elif self.half and not self.first:
            self._getitem = lambda idx : (self.data[idx, :, :self.half_len],
                                          str(self.paths[idx]))
        else:
            self._getitem = lambda idx : (self.data[idx, :, :],
                                          str(self.paths[idx]))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self._getitem(idx)

    @classmethod
    def prepare_data(cls, model, hparams):
        if model.ds is None:
            print('Loading the i3d data from disk. This can take '
                  'several minutes...', flush=True)
        model.ds = model.ds or cls()
        model.ds_length = len(model.ds)
        
        n_test, n_val = hparams.n_test, hparams.n_val
        indices = list(range(model.ds_length))
        
        model.test_sampler = SubsetRandomSampler(indices[:n_test])
        model.val_sampler = SubsetRandomSampler(indices[n_test : n_test+n_val])
        model.train_sampler = SubsetRandomSampler(indices[n_test+n_val:])

    @staticmethod
    def train_dataloader(model, hparams):
        return DataLoader(model.ds, 
                          batch_size=model.batch_size, 
                          sampler=model.train_sampler,
                          num_workers=hparams.n_workers)

    @staticmethod
    def val_dataloader(model, hparams):
        return DataLoader(model.ds, 
                          batch_size=model.batch_size, 
                          sampler=model.val_sampler,
                          num_workers=hparams.n_workers)

        
class Breakfast64DimFVDataset(_BreakfastClipsDataset):
    def __init__(self, *args, **kwargs):
        # Data Paths
        data_names_path = [index.PATH_BK_64_FVS_EVENT_PATHS,
                           index.PATH_BK_64_FVS_NONEVENT_PATHS]
        data_path = [index.PATH_BK_64_FVS_EVENT_DATA,
                     index.PATH_BK_64_FVS_NONEVENT_DATA]
        self.fv_len = 64
        super().__init__(data_names_path, data_path, *args, **kwargs)


class BreakfastI3DFVDataset(_BreakfastClipsDataset):
    def __init__(self, *args, **kwargs):
        # Data Paths
        data_names_path = [index.PATH_BK_I3D_FVS_EVENT_PATHS,
                           index.PATH_BK_I3D_FVS_NONEVENT_PATHS]
        data_path = [index.PATH_BK_I3D_FVS_EVENT_DATA,
                     index.PATH_BK_I3D_FVS_NONEVENT_DATA]
        self.fv_len = 2048
        super().__init__(data_names_path, data_path, *args, **kwargs)
        

class FlattenedImageDataset(IterableDataset):
    def __init__(self, dir_root=index.DIR_BK_VIDEOS, glob='*.avi'):

        self.dir_root = Path(dir_root)
        self.glob = glob
    
        assert self.dir_root.exists() # Quick check

        # How to loop through passed paths
        if self.dir_root.is_dir():
            self.paths_all = self.dir_root.rglob(self.glob)
        elif self.dir_root.is_file():
            self.paths_all = [self.dir_root]
        else:
            raise Exception

    def next_frame(self):
        """Returns the path, frame number, and tensor"""
        # Loop through each path
        for path in self.paths_all:
            # Load the associated video
            cap = cv2.VideoCapture(str(path))

            # While we are still getting frames for this video
            i = 0
            ret = True
            while ret:
                ret, frame = cap.read()
                frame = torch.from_numpy(frame)
                # HWC2CHW
                frame = frame.permute(2, 0, 1)
                yield str(path), i, frame
                i += 1

    def __iter__(self):
        #Create an iterator
        return self.next_frame()
        

class BreakfastImageDataset(IterableDataset):

    def __init__(self, path_root=index.DIR_BK_VIDEOS, glob='*.avi',
                 time_depth=DEFAULT_BATCH_SIZE, retries=1):

        self.path_root = Path(path_root)
        self.glob = glob
        self.time_depth = time_depth

        assert self.path_root.exists() # Quick check

        # How to loop through passed paths
        if self.path_root.is_dir():
            self.paths_all = self.path_root.rglob(self.glob)
        elif self.path_root.is_file():
            self.paths_all = [self.path_root]
        else:
            raise Exception

        #And that's it, we no longer need to store the contents in the memory

    def get_batch(self):
        # Loop through each path
        for path in self.paths_all:
            # Load the associated video
            cap = cv2.VideoCapture(str(path))

            # While we are still getting frames for this video
            ret = True
            while ret:
                frames = torch.FloatTensor(
                    self.time_depth, DIM_C, DIM_H, DIM_W
                )

                # Go through and grab the right number of frames
                for i in range(self.time_depth):      
                    ret, frame = cap.read()
                    # If still couldnt get the frame, exit
                    if not ret:
                        frames = frames[:i, :, :, :] # Unclear if this will work
                        break

                    frame = torch.from_numpy(frame)
                    # HWC2CHW
                    frame = frame.permute(2, 0, 1)
                    frames[i, :, :, :] = frame

                yield frames
                
        
    def __iter__(self):
        #Create an iterator
        return self.get_batch()

# What probably want is just to do batching in dl as the way to choose t
# Unclear how to handle the end of one file and moving to the next
# # Maybe a new video index that is none if there isn't a new video?
