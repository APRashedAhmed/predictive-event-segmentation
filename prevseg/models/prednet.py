"""Basic prednet"""
import time
import logging
from functools import wraps

import torch
import torch.nn as nn
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
    def __init__(self, parent, layer_num, hparams, a_channels, r_channels):
        super().__init__()
        self.parent = parent
        self.layer_num = layer_num        
        if isinstance(hparams, dict):
            self.hparams = Namespace(**hparams)
        else:
            self.hparams = hparams
        self.a_channels = a_channels
        self.r_channels = r_channels
        
        # Reccurent
        self.recurrent = LSTM(2 * self.a_channels[self.layer_num],
                              self.r_channels[self.layer_num])
        self.recurrent.reset_parameters()
        
        # Dense
        self.dense = nn.Sequential(
            nn.Linear(self.r_channels[self.layer_num],
                      self.a_channels[self.layer_num]),
            nn.ReLU())
        if self.layer_num == 0:
            self.dense.add_module('satlu', SatLU())
            
        # Update
        if self.layer_num < self.hparams.n_layers - 1:
            self.update_a = nn.Sequential(
                nn.Linear(
                    2 * self.a_channels[self.layer_num],
                    self.a_channels[self.layer_num + 1]),
                nn.ReLU())
        
        # Build E, R, and H
        self.reset()
        
        # Book keeping
        self.modules = {'recurrent' : self.recurrent, 'dense' : self.dense}
        if hasattr(self, 'update_a'):
            self.modules['update_a'] = self.update_a
        # Hack to appease the pytorch-gods
        for name, module in self.modules.items():
            setattr(self.parent, f'predcell_{self.layer_num}_{name}', module)
            
    def reset(self, batch_size=None):
        batch_size = batch_size or self.hparams.batch_size
        # E, R, and H variables
        self.E = Variable(torch.zeros(
            1,                  # Single time step
            batch_size,
            2 * self.a_channels[self.layer_num])).cuda()
        self.R = Variable(torch.zeros(
            1,                  # Single time step
            batch_size,
            self.r_channels[self.layer_num])).cuda()
        self.H = None

        
class PredNet(pl.LightningModule):
    def __init__(self, hparams, ds=None):
        super().__init__()
        # Attribute definitions
        self.hparams = hparams
        self.n_layers = self.hparams.n_layers
        self.output_mode = self.hparams.output_mode
        self.input_size = self.hparams.input_size
        self.time_steps = self.hparams.time_steps
        self.batch_size = self.hparams.batch_size
        self.ds = ds

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
        self.predcells = [PredCell(self,
                                   layer_num,
                                   self.hparams,
                                   self.a_channels,
                                   self.r_channels)
                          for layer_num in range(self.n_layers)]
        
        # How to weight the errors
        # 1 followed by zeros means just minimize error at lowest layer
        self.layer_loss_weights = Variable(torch.FloatTensor(
            [[1.]] + [[0.]]*(self.n_layers-1)).cuda())
        # How much to weight errors at each timestep
        self.time_loss_weights = 1. / (self.time_steps - 1) \
                                 * torch.ones(self.time_steps, 1)
        # Dont count first time step
        self.time_loss_weights[0] = 0
        self.time_loss_weights = Variable(self.time_loss_weights.cuda())
        
        if self.hparams.device == 'cuda' and torch.cuda.is_available():
            print('Using GPU', flush=True)
            self.cuda()

    def forward(self, input):
        total_error = []
        # Set the expected batch size
        for cell in self.predcells:
            cell.reset(input.size(0))

        for t in range(self.time_steps):
            A = input[:,t,:].unsqueeze(0)
            A = A.type(torch.cuda.FloatTensor)

            # Loop backwards
            for cell in reversed(self.predcells):
                E, R = cell.E, cell.R
                # First time step
                if t == 0:
                    hx = (R, R)
                else:
                    hx = cell.H

                cell.R, cell.H = cell.recurrent(E, hx)

            for cell in self.predcells:
                # Go from R to A_hat
                A_hat = cell.dense(cell.R)

                # Convenience
                if cell.layer_num == 0:
                    frame_prediction = A_hat

                # Split to 2 Es
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg], 2)
                cell.E = E

                # If not last layer, update stored A
                if cell.layer_num < self.n_layers - 1:
                    A = cell.update_a(E)

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
            return frame_prediction

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
        data = Variable(data)
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
