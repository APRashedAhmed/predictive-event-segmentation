"""Basic prednet"""
import time
import logging
import argparse
from functools import wraps

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.nn import functional as F, GRU
from torch.autograd import Variable
from pytorch_lightning.core.decorators import auto_move_data

import prevseg.constants as const
from prevseg import datasets
from prevseg.utils import child_argparser
from prevseg.torch.lstm_cell import LSTM
from prevseg.visualization import visualize_schapiro_walk

logger = logging.getLogger(__name__)


class PredCell(object):
    name = 'predcell'
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
        dense = nn.Sequential(nn.Linear(self.r_channels[self.layer_num],
                                        self.a_channels[self.layer_num]))
        if self.layer_num == 0:
            dense.add_module('tanh', nn.Tanh())
        else:
            dense.add_module('relu', nn.ReLU())
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

    def init_diff_lists(self):
        # Tracking
        if self.parent is not None:
            for tracked_attr in self.parent.track:
                setattr(self, f'{tracked_attr}_diff_list', list())        
            
    def reset(self, batch_size=None):
        batch_size = batch_size or self.hparams.batch_size
        self.init_diff_lists()
        # E, R, and H variables
        self.E = torch.zeros(1,                  # Single time step
                             batch_size,
                             2*self.a_channels[self.layer_num],
                             device=self.parent.dev)
        self.R = torch.zeros(1,                  # Single time step
                             batch_size,
                             self.r_channels[self.layer_num],
                             device=self.parent.dev)
        self.H = (torch.zeros(1,                  # Single time step
                              batch_size,
                              self.r_channels[self.layer_num],
                              device=self.parent.dev),
                  torch.zeros(1,                  # Single time step
                              batch_size,
                              self.r_channels[self.layer_num],
                              device=self.parent.dev))
        
    def update_parent(self, module_names=('recurrent', 'dense', 'update_a')):
        # Hack to appease the pytorch-gods
        for module_name in module_names:
            if hasattr(self, module_name) and \
               getattr(self, module_name) is not None:
                setattr(self.parent,
                        f'{self.name}_{self.layer_num}_{module_name}',
                        getattr(self, module_name))

    def track_metric_diff(self, m1, m2, name, output_mode=None):
        # Steal parent's output mode if there is one and None is passed
        if self.parent is not None and not output_mode:
            output_mode = self.parent.output_mode
            
        if name in self.parent.track and output_mode == 'eval':
            diff = torch.mean(
                (m1.permute(1, 0, 2) - m2.permute(1, 0, 2))**2).detach()
            scalar_name = f'{name}_diff_layer_{self.layer_num}'
            self.parent.logger.experiment.log_metric(scalar_name, diff)
            getattr(self, f'{name}_diff_list').append(diff)                

                
class PredNet(pl.LightningModule):
    name = 'prednet'
    track = ('hidden', 'error', 'representation')
    def __init__(self, hparams, ds=None, a_channels=None, r_channels=None,
                 CellClass=PredCell, ds_val=None, track=None):
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
        self.ds_val = ds_val
        self.a_channels = a_channels
        self.r_channels = r_channels
        self.CellClass = CellClass
        self.track = track or self.track

        if hasattr(datasets, hparams.dataset):
            self.Dataset = getattr(datasets, hparams.dataset)
        else:
            raise Exception(
                f'Invalid dataset "{self.hparams.dataset}" passed. Check it is '
                f'importable: "from prevseg.datasets import '
                f'{self.hparams.dataset}"'
            )
        
        self.dev = 'cuda:0'
        
        # Put together the model
        self.build_model()                
        
    def build_model(self):
        # Channel sizes
        if self.r_channels is None:
            self.r_channels = [self.input_size // (2**i) 
                               for i in range(self.n_layers)] + [0,]
        if self.a_channels is None:
            self.a_channels = [self.input_size // (2**i) 
                               for i in range(self.n_layers)]
        
        # Make sure everything checks out
        default_output_modes = ['prediction', 'error']
        assert self.output_mode in default_output_modes, \
            'Invalid output_mode: ' + str(output_mode)

        # Make all the pred cells
        self.cells = [self.CellClass(self,
                                     layer_num,
                                     self.hparams,
                                     self.a_channels,
                                     self.r_channels)
                      for layer_num in range(self.n_layers)]
        
        # How to weight the errors
        # 1 followed by zeros means just minimize error at lowest layer
        self.layer_loss_weights = self.build_layer_loss_weights(
            self.layer_loss_mode) if self.layer_loss_mode else None
        # How much to weight errors at each timezstep
        self.time_loss_weights = self.build_time_loss_weights()
        
    def build_layer_loss_weights(self, mode='first'):
        if type(mode) is str:
            if mode == 'first':
                first = torch.zeros(self.n_layers, 1, device=self.dev)
                first[0][0] = 1
                return first
            elif mode == 'all':
                return 1. / (self.n_layers-1) * torch.ones(self.n_layers, 1,
                                                           device=self.dev)
            elif mode == 'tri':
                out = 1. / sum(range(1, self.n_layers + 1)) * torch.arange(
                    1, self.n_layers + 1, device=self.dev)
                out = out.unsqueeze(-1)
                return out
            elif mode == 'exp2':
                out = 1. / sum([2**l for l in range(1, self.n_layers+1)]) * \
                    torch.tensor([2**l for l in range(1, self.n_layers+1)],
                                 device=self.dev)
                out = out.unsqueeze(-1)
                return out
            elif mode == 'exp10':
                out = 1. / sum([10**l for l in range(1, self.n_layers+1)]) * \
                    torch.tensor([10**l for l in range(1, self.n_layers+1)],
                                 device=self.dev)
                out = out.unsqueeze(-1)
            else:
                raise Exception(f'Invalid layer loss mode "{mode}".')
        elif isinstance(mode, torch.Tensor):
            return mode
            
    def build_time_loss_weights(self, time_steps=None):
        time_steps = time_steps or self.time_steps
        # How much to weight errors at each timestep
        time_loss_weights = 1. / (time_steps-1) * torch.ones(time_steps, 1,
                                                             device=self.dev)
        # Dont count first time step
        time_loss_weights[0] = 0
        return time_loss_weights
    
    def check_input_shape(self, input):
        batch_size, time_steps, *input_size = input.shape

        for cell in self.cells:
            cell.reset(batch_size)
                
        # Reset time_step-dependent things
        if time_steps != self.time_steps:
            self.time_steps = time_steps
            self.time_loss_weights = self.build_time_loss_weights(
                self.time_steps)
            
        return batch_size, time_steps, *input_size

    def top_down_pass(self):
        # Loop backwards
        for l, cell in reversed(list(enumerate(self.cells))):
            # Convenience
            E, R = cell.E, cell.R
            
            # First time step
            if self.t == 0:
                hx = (R, R)
            else:
                hx = cell.H

            # If not in the last layer, upsample R and
            if l < self.n_layers - 1:
                E = torch.cat((E,  cell.upsample(self.cells[l+1].R)), 2)

            # Update the values of R and H
            R, H = cell.recurrent(E, hx)

            # Optional tracking
            if self.t != 0:
                cell.track_metric_diff(R, cell.R, 'representation')
                cell.track_metric_diff(H[1], cell.H[1], 'hidden') # Just C array

            # Update cell state
            cell.R, cell.H = R, H
            
    def bottom_up_pass(self):
        for l, cell in enumerate(self.cells):
            # Go from R to A_hat
            A_hat = cell.dense(cell.R)

            # Convenience
            if self.output_mode == 'prediction' and l == 0:
                self.frame_prediction = A_hat

            # Split to 2 Es
            pos = F.relu(A_hat - self.A)
            neg = F.relu(self.A - A_hat)
            E = torch.cat([pos, neg], 2)
            
            # Optional Error tracking
            cell.track_metric_diff(E, cell.E, 'error')
            
            # Update cell error
            cell.E = E
 
            # If not last layer, update stored A
            if l < self.n_layers - 1:
                self.A = cell.update_a(E)

    @auto_move_data                
    def forward(self, input, output_mode=None):
        self.output_mode = output_mode or self.output_mode
        _, time_steps, *_ = self.check_input_shape(input)
        
        self.total_error = []
        
        for self.t in range(time_steps):
            self.A = input[:,self.t,:].unsqueeze(0).to(self.dev, torch.float)
            # Loop from top layer to update R and H
            self.top_down_pass()
            # Loop bottom up to get E and A
            self.bottom_up_pass()
            # Track desired outputs
            self.track_outputs()
        
        return self.return_output()

    def track_outputs(self):
        if self.output_mode == 'error':
            mean_error = torch.cat(
                [torch.mean(cell.E.view(cell.E.size(1), -1),
                            1, keepdim=True)
                 for cell in self.cells], 1)
            # batch x n_layers
            self.total_error.append(mean_error)
            
    def return_output(self):
        if self.output_mode == 'error':
            return torch.stack(self.total_error, 2) # batch x n_layers x nt
        elif self.output_mode == 'prediction':
            return self.frame_prediction
        elif self.output_mode == 'eval':
            return self.eval_outputs()

    def eval_outputs(self):
        outputs = {}
        for tracked in self.track:
            outputs[tracked] = torch.cat(
                [torch.Tensor(getattr(cell, tracked+'_diff_list')).unsqueeze(0)
                 for cell in self.cells])
        return outputs
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def _common_step(self, batch, batch_idx):
        data, path = batch
        prediction_errors = self.forward(data) # batch x n_layers x nt
        loc_batch = prediction_errors.size(0)
        loss = torch.mm(prediction_errors.view(-1, self.time_steps), 
                          self.time_loss_weights) # batch*n_layers x 1
        if self.layer_loss_mode is not None:
            loss = torch.mm(loss.view(loc_batch, -1), 
                              self.layer_loss_weights)
        return torch.mean(loss, axis=0)
        
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        result = pl.TrainResult(loss)
        # Unclear why this *doesn't* work, but it doesnt log at every step.
        # Best I got was logging at every epoch
        
        # result.log('loss', loss, prog_bar=True)
        
        # This works though, so going with it for now
        self.logger.experiment.log_metric('loss', loss)
        return result
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True, on_step=False,
                   on_epoch=True)
        return result

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = child_argparser(parent_parser)
        parser.add_argument('--n_layers', type=int, default=2)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--output_mode', type=str, default='error')
        parser.add_argument('--layer_loss_mode', type=str, default='first')

        # See if we have the right number of inputs and n_layers has been
        # specified to infer default batch size
        temp_args, _ = parser.parse_known_args()
        default_batch_size = 256
        if temp_args.input_size is not None and temp_args.input_size <= 2048:
            if temp_args.n_layers is not None:
                if temp_args.n_layers == 2:
                    default_batch_size = 256 + 128
                elif temp_args.n_layers == 1:
                    default_batch_size = 512 + 128 + 64
                
        parser.add_argument('--batch_size', type=int,
                            default=default_batch_size)
        
        return parser
    
    def visualize(self, data, borders=None, offset=0):
        figs = {}
        if 'error' in data.keys():
            figs = {'errors' : visualize_schapiro_walk(
                data['errors'].view(self.hparams.n_layers, -1),
                vlines=borders,
                title='Mean Prediction Errors',
                idx=0)}
        
        for i, diff in enumerate(self.track):
            figs[diff+'_diffs'] = visualize_schapiro_walk(
                data[diff],
                vlines=borders,
                offset=offset,
                title=f'Mean {diff.replace("_", " ").title()}',
                idx=i+1,
            )
        return figs

    # def prepare_data(self, *args, **kwargs):
    #     self.Dataset.prepare_data(self, self.hparams, *args, **kwargs)
        
    # def train_dataloader(self, *args, **kwargs):
    #     return self.Dataset.train_dataloader(self, self.hparams, *args,
    #                                          **kwargs)
    
    # def val_dataloader(self, *args, **kwargs):
    #     return self.Dataset.val_dataloader(self, self.hparams, *args, **kwargs)
