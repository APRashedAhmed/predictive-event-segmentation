"""Script for the Schapiro dataloader"""
import logging
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import IterableDataset

from prevseg import index
from prevseg.utils import isiterable
from prevseg.schapiro import walk, graph

logger = logging.getLogger(__name__)


class ShapiroFractalsDataset(IterableDataset):
    modes = set(('random', 'euclidean', 'hamiltonian', 'custom'))
    def __init__(self, batch_size=32, n_pentagons=3, max_steps=128, n_paths=128,
                 mapping=None, mode='random', debug=False, custom_path=None):
        self.batch_size = batch_size
        self.n_pentagons = n_pentagons
        self.max_steps = max_steps
        self.n_paths = n_paths
        self.mapping = mapping
        self.mode = mode
        self.debug = debug
        self.custom_path = custom_path
        assert self.mode in self.modes
        if self.mode == 'custom':
            assert self.custom_path is not None and isiterable(self.custom_path)
            assert mapping is not None
            self.batch_size = 1
            self.n_paths = 1
            self.max_steps = len(self.custom_path)
        
        self.G = graph.schapiro_graph(n_pentagons=n_pentagons)
        
        self.load_node_stimuli()
        
        self.mapping = {node : path.stem
                        for node, path in zip(range(len(self.G.nodes)),
                                              self.paths_data)}
        print(f'Created mapping as follows:\n{self.mapping}')
        
        if self.debug:
            self.sample_transform = lambda sample : sample
        else:
            self.sample_transform = lambda sample : self.array_data[sample]
        super().__init__()
        
    def load_node_stimuli(self):
        # Load the fractal images into memory
        assert index.DIR_SCH_FRACTALS.exists()
        if self.mapping:
            self.paths_data = [index.DIR_SCH_FRACTALS / (name+'.tiff')
                               for name in self.mapping.values()]
        else:
            paths_data = list(index.DIR_SCH_FRACTALS.iterdir())
            np.random.shuffle(paths_data)
            self.paths_data = paths_data[:5*self.n_pentagons]
        self.array_data = np.array(
            [np.array(ImageOps.grayscale(Image.open(str(path))))
             for path in self.paths_data])
        
    def iter_single_sample(self): 
        if self.mode == 'random':
            iter_walk = walk.walk_random(self.G, steps=self.max_steps)
        elif self.mode == 'euclidean':
            iter_walk = walk.walk_euclidean(self.G)
        elif self.mode == 'hamiltonian':
            iter_walk = walk.walk_hamiltonian(self.G)
        elif self.mode == 'custom':
            iter_walk = zip([self.custom_path, self.custom_path]) # match api

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
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument('--input_size', type=int, default=3*128*160)
        parser.add_argument('--time_steps', type=int, default=128)
        parser.add_argument('--max_steps', type=int, default=128)
        parser.add_argument('--n_paths', type=int, default=16)
        parser.add_argument('--n_pentagons', type=int, default=3)
        return parser

    
class ShapiroResnetEmbeddingDataset(ShapiroFractalsDataset):
    def load_node_stimuli(self):
        # Load the fractal images into memory
        assert index.DIR_SCH_FRACTALS.exists()
        if self.mapping:
            self.paths_data = [index.DIR_SCH_FRACTALS_EMB / (name+'.npy')
                               for name in self.mapping.values()]
        else:
            paths_data = list(index.DIR_SCH_FRACTALS_EMB.iterdir())
            np.random.shuffle(paths_data)
            self.paths_data = paths_data[:5*self.n_pentagons]
        self.array_data = np.array(
            [np.array(np.load(str(path)))
             for path in self.paths_data])    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_cls_parser = ShapiroFractalsDataset.add_model_specific_args(
            parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_cls_parser],
                                         add_help=False,
                                         conflict_handler='resolve')
        parser.add_argument('--input_size', type=int, default=2048)
        return parser
