"""Script for the Schapiro dataloader"""
import logging
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, IterableDataset

from prevseg import index
from prevseg.utils import isiterable, child_argparser
from prevseg.schapiro import walk, graph

logger = logging.getLogger(__name__)


class SchapiroFractalsDataset(IterableDataset):
    modes = set(('random', 'euclidean', 'hamiltonian', 'custom'))
    def __init__(self, batch_size=32, n_pentagons=3, max_steps=128, n_paths=128,
                 mapping=None, mode='random', debug=False, custom_path=None,
                 dir_data=index.DIR_SCH_FRACTALS, verbose=False):            
        super().__init__()
        self.batch_size = batch_size
        self.n_pentagons = n_pentagons
        self.max_steps = max_steps
        self.n_paths = n_paths
        self.mode = mode
        self.debug = debug
        self.custom_path = custom_path
        self.verbose = True if self.debug else verbose

        self.dir_data = dir_data
        assert self.dir_data.exists()        

        assert self.mode in self.modes
        if self.mode == 'custom':
            assert self.custom_path is not None and isiterable(self.custom_path)
            assert mapping is not None # Mapping the input
            self.batch_size = 1
            self.n_paths = 1
            self.max_steps = len(self.custom_path)

        self.G = graph.schapiro_graph(n_pentagons=n_pentagons)            

        self.mapping = mapping or self.random_mapping(self.dir_data,
                                                      self.n_pentagons,
                                                      self.G)
        if self.verbose:
            print(f'Created mapping as follows:\n{self.mapping}')
        
        self.array_data = self.load_node_stimuli()
        
        if self.debug:
            self.sample_transform = lambda sample : sample
        else:
            self.sample_transform = lambda sample : self.array_data[sample]
        
    def load_node_stimuli(self, suffix=None):
        suffix = suffix or '.tiff'
        # Load the fractal images into memory            
        return {node : ImageOps.grayscale(Image.open(
            str(self.dir_data/(name + suffix))))
                for node, name in self.mapping.items()
        }
        
    def iter_single_sample(self): 
        if self.mode == 'random':
            iter_walk = walk.walk_random(self.G, steps=self.max_steps)
        elif self.mode == 'euclidean':
            iter_walk = walk.walk_euclidean(self.G)
        elif self.mode == 'hamiltonian':
            iter_walk = walk.walk_hamiltonian(self.G)
        elif self.mode == 'custom':
            iter_walk = [[s] for s in self.custom_path]

        for sample in iter_walk:
            yield self.array_data[sample[0]], sample[0]
        
    def iter_batch_sample(self):
        iter_batch = zip(*[self.iter_single_sample()
                           for _ in range(self.batch_size)])
        for batch in iter_batch:
            data, nodes = zip(*batch)
            yield data, nodes
        
    def iter_batch_dataset(self):   
        for _ in range(self.n_paths):
            data, nodes = zip(*list(self.iter_batch_sample()))
            yield np.moveaxis(np.array(data), 0, 1), nodes
        
    def __iter__(self):
        return self.iter_batch_dataset()
    
    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument('--input_size', type=int, default=3*128*160)
        parser.add_argument('--time_steps', type=int, default=128)
        parser.add_argument('--max_steps', type=int, default=128)
        parser.add_argument('--n_paths', type=int, default=10)
        parser.add_argument('--n_pentagons', type=int, default=3)
        return parser

    @staticmethod
    def random_mapping(dir_data=index.DIR_SCH_FRACTALS, n_pentagons=3, G=None):
        dir_data = Path(dir_data)
        paths_all_data = list(dir_data.iterdir())
        np.random.shuffle(paths_all_data)
        paths_mapping_data = paths_all_data[:5*n_pentagons]
        
        if not G:
            G = graph.schapiro_graph(n_pentagons=n_pentagons)
        
        return {node : path.stem
                for node, path in zip(range(len(G.nodes)),
                                      paths_mapping_data)}

    @classmethod
    def prepare_data(cls, datamodule, val=False):
        pass
    
    @classmethod
    def setup_data(cls, hparams, datamodule, val=False):
        datamodule.ds = datamodule.ds or cls(
            batch_size=hparams.batch_size,
            n_pentagons=hparams.n_pentagons, 
            max_steps=hparams.max_steps, 
            n_paths=hparams.n_paths,
            mapping=eval(hparams.mapping), # This is a str in hparams
        )
        datamodule.ds_val = datamodule.ds_val or cls(
            n_pentagons=hparams.n_pentagons,
            batch_size=hparams.batch_size, 
            n_paths=hparams.n_val,
            max_steps=hparams.max_steps, 
            mapping=eval(hparams.mapping),
            mode='custom' if hparams.val_path else 'euclidean',
            custom_path=eval(hparams.val_path),
        )

    @staticmethod
    def train_dataloader(hparams, datamodule, ds):
        return DataLoader(ds, 
                          batch_size=None,
                          num_workers=hparams.n_workers)
    
    @staticmethod
    def val_dataloader(hparams, datamodule, ds_val):
        return DataLoader(ds_val, 
                          batch_size=None,
                          num_workers=hparams.n_workers)
    
    
class SchapiroResnetEmbeddingDataset(SchapiroFractalsDataset):
    def __init__(self, dir_data=index.DIR_SCH_FRACTALS_EMB, *args, **kwargs):
        super().__init__(dir_data=dir_data, *args, **kwargs)

    def load_node_stimuli(self, suffix=None):
        suffix = suffix or '.npy'
        # Load the fractal images into memory            
        return {node : np.load(str(self.dir_data/(name + suffix)))
                for node, name in self.mapping.items()}

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = child_argparser(
            SchapiroFractalsDataset.add_dataset_specific_args(parent_parser))
        parser.add_argument('--input_size', type=int, default=2048)
        return parser
