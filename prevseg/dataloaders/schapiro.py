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
    def __init__(self, batch_size=32, n_pentagons=3, max_steps=1024):
        self.batch_size = batch_size
        self.n_pentagons = n_pentagons
        self.max_steps = max_steps
        
        self.G = graph.schapiro_graph(n_pentagons=n_pentagons)
        
        self.load_node_stimuli()
        
        self.mapping = {node : path.stem
                        for node, path in zip(range(len(self.G.nodes)),
                                              self.paths_data)}
        print(f'Created mapping as follows:\n{self.mapping}')
        
    def load_node_stimuli(self):
        # Load the fractal images into memory
        assert index.DIR_SCH_FRACTALS.exists()
        paths_data = list(index.DIR_SCH_FRACTALS.iterdir())
        np.random.shuffle(paths_data)
        self.paths_data = paths_data[:5*self.n_pentagons]
        self.array_data = np.array(
            [np.array(ImageOps.grayscale(Image.open(str(path))))
             for path in self.paths_data])
        
    def iter_single_sample(self):
        for sample in walk.walk_random(self.G, steps=self.max_steps):
            yield self.array_data[sample[0]]
        
    def iter_batch_dataset(self):
        batch = zip(*[self.iter_single_sample() for _ in range(self.batch_size)])
        try:
            while True:
                yield np.array(next(batch))
        except StopIteration:
            return
        
    def __iter__(self):
        return self.iter_batch_dataset()


class ShapiroResnetEmbeddingDataset(ShapiroFractalsDataset):
    def load_node_stimuli(self):
        # Load the fractal images into memory
        assert index.DIR_SCH_FRACTALS.exists()
        paths_data = list(index.DIR_SCH_FRACTALS_EMB.iterdir())
        np.random.shuffle(paths_data)
        self.paths_data = paths_data[:5*self.n_pentagons]
        self.array_data = np.array(
            [np.array(np.load(str(path)))
             for path in self.paths_data])    
