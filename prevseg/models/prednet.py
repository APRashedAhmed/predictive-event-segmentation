"""Basic prednet"""
import time
import logging
from functools import wraps

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.nn import functional as F, GRU
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from prevseg.torch.lstm import LSTM
from prevseg.torch.activations import SatLU

logger = logging.getLogger(__name__)


class PredCell(object):
    """Organizational class."""
    def __init__(self, parent, layer_num, hparams, a_channels, r_channels, 
                 RecurrentClass=LSTM):
        super().__init__()
        self.parent = parent
        self.layer_num = layer_num
        self.hparams = hparams
        self.a_channels = a_channels
        self.r_channels = r_channels
        self.RecurrentClass = RecurrentClass
        
        # Reccurent
        self.recurrent = self.build_recurrent()
        # Dense
        self.dense = self.build_dense()
        # Update
        self.update_a = self.build_update()
        # upsample - set at cell level for future
        self.upsample = nn.Upsample(scale_factor=2)
        
        # Build E, R, and H
        self.reset()
        # Book-keeping
        self.update_parent()
            
    def build_recurrent(self):
        recurrent = self.RecurrentClass(
            2 * (self.a_channels[self.layer_num] +
                 self.r_channels[self.layer_num+1]),
            #+ self.r_channels[self.layer_num+1],
            self.r_channels[self.layer_num])
        recurrent.reset_parameters()
        return recurrent
    
    def build_dense(self):
        dense = nn.Sequential(
            nn.Linear(self.r_channels[self.layer_num],
                      self.a_channels[self.layer_num]),
            nn.ReLU())
        if self.layer_num == 0:
            dense.add_module('satlu', SatLU())
        return dense
        
    def build_update(self):
        if self.layer_num < self.hparams.n_layers - 1:
            return nn.Sequential(
                nn.Linear(
                    2 * self.a_channels[self.layer_num],
                    self.a_channels[self.layer_num + 1]),
                nn.ReLU())
        else:
            return None
            
    def reset(self, batch_size=None):
        batch_size = batch_size or self.hparams.batch_size
        # E, R, and H variables
        self.E = torch.zeros(1,                  # Single time step
                             batch_size,
                             2*self.a_channels[self.layer_num],
                             device=self.parent.device)
        self.R = torch.zeros(1,                  # Single time step
                             batch_size,
                             self.r_channels[self.layer_num],
                             device=self.parent.device)
        self.H = None
        
    def update_parent(self):
        self.modules = {'recurrent' : self.recurrent, 'dense' : self.dense}
        if hasattr(self, 'update_a') and self.update_a is not None:
            self.modules['update_a'] = self.update_a
        # Hack to appease the pytorch-gods
        for name, module in self.modules.items():
            setattr(self.parent, f'predcell_{self.layer_num}_{name}', module)


class PredNet(pl.LightningModule):
    name = 'prednet'
    def __init__(self, hparams, ds=None, CellClass=PredCell):
        super().__init__()
        # Attribute definitions
        self.hparams = hparams
        self.n_layers = self.hparams.n_layers
        self.output_mode = self.hparams.output_mode
        self.input_size = self.hparams.input_size
        self.time_steps = self.hparams.time_steps
        self.batch_size = self.hparams.batch_size
        self.layer_loss_mode = self.hparams.layer_loss_mode
        self.ds = ds
        self.CellClass = CellClass
        
        if self.hparams.device == 'cuda' and torch.cuda.is_available():
            print('Using GPU', flush=True)
            self.device = torch.device('cuda')
        else:
            print('Using CPU', flush=True)
            self.device = torch.device('cpu')

        # Put together the model
        self.build_model()

    def build_model(self):        
        # Channel sizes
        self.r_channels = [self.input_size // (2**i) 
                           for i in range(self.n_layers)] + [0,] # Convenience
        self.a_channels = [self.input_size // (2**i) 
                           for i in range(self.n_layers)]
        
        # Make sure everything checks out
        default_output_modes = ['prediction', 'error']
        assert self.output_mode in default_output_modes, \
            'Invalid output_mode: ' + str(output_mode)

        # Make all the pred cells
        self.predcells = [self.CellClass(self,
                                         layer_num,
                                         self.hparams,
                                         self.a_channels,
                                         self.r_channels)
                          for layer_num in range(self.n_layers)]
        
        # How to weight the errors
        # 1 followed by zeros means just minimize error at lowest layer
        self.layer_loss_weights = self.build_layer_loss_weights(
            self.layer_loss_mode)
        # How much to weight errors at each timestep
        self.time_loss_weights = self.build_time_loss_weights()
        
    def build_layer_loss_weights(self, mode='first'):
        if mode == 'first':
            first = torch.zeros(self.n_layers, 1, device=self.device)
            first[0][0] = 1
            return first
        elif mode == 'all':
            return 1. / (self.n_layer-1) * torch.ones(self.n_layer, 1,
                                                      device=self.device)
        else:
            raise Exception(f'Invalid layer loss mode "{mode}".')
            
    def build_time_loss_weights(self, time_steps=None):
        time_steps = time_steps or self.time_steps
        # How much to weight errors at each timestep
        time_loss_weights = 1. / (time_steps-1) * torch.ones(time_steps, 1,
                                                             device=self.device)
        # Dont count first time step
        time_loss_weights[0] = 0
        return time_loss_weights
    
    def check_input_shape(self, input):
        batch_size, time_steps, *input_size = input.shape
        
        # Reset batch_size-dependent things
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            for cell in self.predcells:
                cell.reset(self.batch_size)
                
        # Reset time_step-dependent things
        if time_steps != self.time_steps:
            self.time_steps = time_steps
            self.time_loss_weights = self.build_time_loss_weights(
                self.time_steps)
            
        return batch_size, time_steps, *input_size
    
    def top_down_pass(self, t):
        # Loop backwards
        for l, cell in reversed(list(enumerate(self.predcells))):
            E, R = cell.E, cell.R
            # First time step
            if t == 0:
                hx = (R, R)
            else:
                hx = cell.H

            # If not in the last layer, upsample R and
            if l < self.n_layers - 1:
                E = torch.cat((E,  cell.upsample(self.predcells[l+1].R)), 2)

            cell.R, cell.H = cell.recurrent(E, hx)
            
    def bottom_up_pass(self):
        for cell in self.predcells:
            # Go from R to A_hat
            A_hat = cell.dense(cell.R)

            # Convenience
            if self.output_mode == 'prediction' and cell.layer_num == 0:
                self.frame_prediction = A_hat

            # Split to 2 Es
            pos = F.relu(A_hat - self.A)
            neg = F.relu(self.A - A_hat)
            E = torch.cat([pos, neg], 2)
            cell.E = E

            # If not last layer, update stored A
            if cell.layer_num < self.n_layers - 1:
                self.A = cell.update_a(E)
            
    def forward(self, input):
        _, time_steps, *_ = self.check_input_shape(input)
        
        total_error = []

        for t in range(time_steps):
            self.A = input[:,t,:].unsqueeze(0).to(self.device, torch.float)
            
            # Loop from top layer to update R and H
            self.top_down_pass(t)
            # Loop bottom up to get E and A
            self.bottom_up_pass()
            
            if self.output_mode == 'error':
                mean_error = torch.cat(
                    [torch.mean(cell.E.view(cell.E.size(1), -1),
                                1, keepdim=True)
                     for cell in self.predcells], 1)
                # batch x n_layers
                total_error.append(mean_error)
        
        if self.output_mode == 'error':
            return torch.stack(total_error, 2) # batch x n_layers x nt
        elif self.output_mode == 'prediction':
            return self.frame_prediction

    def timeit(method):
        """Combination of https://stackoverflow.com/questions/51503672/decorator-for-timeit-timeit-method/51503837#51503837,
        and https://www.geeksforgeeks.org/python-program-to-convert-seconds-into-hours-minutes-and-seconds/"""
        @wraps(method)
        def _time_it(self, *args, **kwargs):
            start = int(round(time.time() * 1000))
            try:
                return method(self, *args, **kwargs)
            finally:
                end_ = int(round(time.time() * 1000)) - start
                if end_ > 1000:
                    time_str = time.strftime("%H:%M:%S",
                                             time.gmtime(end_ // 1000))
                    print(f"Total execution time: {time_str}", flush=True)
                
        return _time_it

    @timeit
    def prepare_data(self):
        if self.ds is None:
            print('Loading the i3d data from disk. This can take '
                  'several minutes...', flush=True)
        self.ds = self.ds or BreakfastI3DFVDataset()
        self.ds_length = len(self.ds)
        np.random.seed(self.hparams.seed)
        self.indices = list(range(self.ds_length))
        self.train_sampler = SubsetRandomSampler(
            self.indices[self.hparams.n_val:])
        self.val_sampler = SubsetRandomSampler(
            self.indices[:self.hparams.n_val])
        
    def train_dataloader(self):
        return DataLoader(self.ds, 
                          batch_size=self.batch_size, 
                          sampler=self.train_sampler,
                          num_workers=self.hparams.n_workers)
    
    def val_dataloader(self):
        return DataLoader(self.ds, 
                          batch_size=self.batch_size, 
                          sampler=self.val_sampler,
                          num_workers=self.hparams.n_workers)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def _common_step(self, batch, batch_idx, mode):
        data, path = batch
        errors = self.forward(data) # batch x n_layers x nt
        loc_batch = errors.size(0)
        errors = torch.mm(errors.view(-1, self.time_steps), 
                          self.time_loss_weights) # batch*n_layers x 1
        errors = torch.mm(errors.view(loc_batch, -1), 
                          self.layer_loss_weights)
        errors = torch.mean(errors, axis=0)
        
        if mode == 'train':
            prefix = ''
        else:
            prefix = mode + '_'
            
        self.logger.experiment.add_scalar(f'{prefix}loss', 
                                          errors, self.global_step)
        return {f'{prefix}loss' : errors}

    def validation_epoch_end(self, output):
        out_dict = {}
        out_dict['val_loss'] = np.mean([out['val_loss'].item()
                                        for out in output])
        out_dict['global_step'] = self.global_step
        return out_dict
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')
