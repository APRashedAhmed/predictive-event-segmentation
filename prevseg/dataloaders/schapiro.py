"""Script for the Schapiro dataloader"""
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import IterableDataset

import prevseg.index as index
from prevseg.schapiro import walk, graph

logger = logging.getLogger(__name__)


class ShapiroFractalsDataset(IterableDataset):
    modes = set(('random', 'euclidean', 'hamiltonian'))
    def __init__(self, batch_size=32, n_pentagons=3, max_steps=128, n_paths=128,
                 mapping=None, mode='random', debug=False):
        self.batch_size = batch_size
        self.n_pentagons = n_pentagons
        self.max_steps = max_steps
        self.n_paths = n_paths
        self.mapping = mapping
        self.mode = mode
        self.debug = debug
        assert self.mode in self.modes
        
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
        
        for sample in iter_walk:
            yield self.sample_transform(sample[0])
        
    def iter_batch_sample(self):
        yield from zip(*[self.iter_single_sample()
                         for _ in range(self.batch_size)])
        
    def iter_batch_dataset(self):       
        for _ in range(self.n_paths):
            yield np.moveaxis(np.array(list(self.iter_batch_sample())), 0, 1)
        
    def __iter__(self):
        return self.iter_batch_dataset()


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
