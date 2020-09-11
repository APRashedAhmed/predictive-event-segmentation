"""Place for 'vanilla' LSTM models"""
import logging
from argparse import Namespace

import torch
import torch.nn as nn
from pytorch_lightning.core.decorators import auto_move_data

import prevseg.models.prednet as pn
import prevseg.constants as const
from prevseg.utils import child_argparser

logger = logging.getLogger(__name__)

class LSTMCell(pn.PredCellTracked):
    name = 'lstmcell'
    
    def __init__(self, parent, layer_num, hparams, a_channels, r_channels,
                 *args, **kwargs):
        self.build_dense = lambda *args, **kwargs : None
        self.build_update = lambda *args, **kwargs : None
        super().__init__(parent, layer_num, hparams, a_channels, r_channels,
                         *args, **kwargs)
        
    def build_recurrent(self):
        recurrent = self.RecurrentClass(
            self.a_channels[self.layer_num],
            #+ self.r_channels[self.layer_num+1],
            self.r_channels[self.layer_num])
        recurrent.reset_parameters()
        return recurrent
        
    def reset(self, batch_size=None):
        batch_size = batch_size or self.hparams.batch_size
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
        self.hidden_full_list = []
        self.hidden_diff_list = []
        self.representation_full_list = []
        self.representation_diff_list = []        
        
    def update_parent(self, module_names=('recurrent',)):
        return super().update_parent(module_names=module_names)


class LSTMCellDense(LSTMCell, pn.PredCellTracked):
    name = 'lstmcell_dense'
    def __init__(self, parent, layer_num, hparams, a_channels, r_channels,
                 *args, **kwargs):
        self.build_update = lambda *args, **kwargs : None
        super(LSTMCell, self).__init__(
            parent, layer_num, hparams, a_channels, r_channels, *args, **kwargs)

    def build_dense(self):
        dense = nn.Sequential(nn.Linear(self.r_channels[self.layer_num],
                                        self.a_channels[self.layer_num]))
        if self.layer_num == self.hparams.n_layers-1:
            dense.add_module('tanh', nn.Tanh())
        else:
            dense.add_module('relu', nn.ReLU())
        return dense        

    def update_parent(self, module_names=('recurrent', 'dense')):
        return super().update_parent(module_names=module_names)


class LSTMStacked(pn.PredNetTrackedSchapiro):
    name = 'lstmstacked'
    track = ('representation_diff', 'hidden_diff')
    def __init__(self, hparams, CellClass=LSTMCell, a_channels=None,
                 r_channels=None, *args, **kwargs):
        if not isinstance(hparams, Namespace):
            hparams = Namespace(**hparams)
        # Assertions for how it should be used
        hparams.layer_loss_mode = None
        
        if a_channels is None:
            a_channels = [hparams.input_size] * hparams.n_layers
        if r_channels is None:
            r_channels = list(a_channels) + [0,]
            
        # Run the init and cleanup
        super().__init__(hparams=hparams, CellClass=CellClass,
                         r_channels=r_channels, a_channels=a_channels, *args,
                         **kwargs)
        
        # Add the last dense layer
        self.dense = self.build_dense(self.r_channels[hparams.n_layers - 1],
                                      self.a_channels[0])
        
        # if self.dense and torch.__version__ == '1.4.0' and torch.cuda.is_available():
        #     self.dense = self.dense.cuda()

    def build_dense(self, r_channels, a_channels):
        return nn.Sequential(nn.Linear(r_channels, a_channels), nn.Tanh())

    @auto_move_data
    def forward(self, input, output_mode=None, track=None, run_num=None,
                tb_labels=None):
        self.run_num = run_num or self.run_num
        self.tb_labels = tb_labels or self.tb_labels
        self.output_mode = output_mode or self.output_mode
        _, time_steps, *_ = self.check_input_shape(input)
        
        total_output = []

        for self.t in range(time_steps):
            self.frame = input[:,self.t,:].unsqueeze(0).to(self.dev,
                                                           torch.float)
            A = self.frame
            for cell in self.cells:
                # First time step
                if self.t == 0:
                    hx = (cell.R, cell.R)
                else:
                    hx = cell.H
                    
                cell.R, cell.H = cell.recurrent(A, hx)
                # Optional tracking
                cell.track_hidden(self.output_mode, hx)
                cell.track_representation(self.output_mode, A)
                A = cell.R
                
            A_hat = self.dense(A)
            pos = nn.functional.relu(A_hat - self.frame)
            neg = nn.functional.relu(self.frame - A_hat)
            E = torch.cat([pos, neg], 2)
            
            if self.output_mode == 'error':
                total_output.append(E)
            elif self.output_mode == 'prediction':
                total_output.append(A_hat)
        
        if self.output_mode == 'eval':
            return self.eval_outputs()
        else:
            return torch.stack(total_output, 2)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = child_argparser(pn.PredNet.add_model_specific_args(
            parent_parser))
        parser.add_argument('--layer_loss_mode', type=str, default='')
        return parser
        

class LSTMStackedDense(LSTMStacked):
    name = 'lstmstacked_dense'
    def __init__(self, hparams, CellClass=LSTMCellDense, a_channels=None,
                 r_channels=None, *args, **kwargs):
        self.build_dense = lambda *args, **kwargs : None
        # Run the init and cleanup
        super().__init__(hparams=hparams, CellClass=CellClass,
                         r_channels=r_channels, a_channels=a_channels, *args,
                         **kwargs)        

    @auto_move_data
    def forward(self, input, output_mode=None, track=None, run_num=None,
                tb_labels=None):
        self.run_num = run_num or self.run_num
        self.tb_labels = tb_labels or self.tb_labels
        self.output_mode = output_mode or self.output_mode
        _, time_steps, *_ = self.check_input_shape(input)
        
        total_output = []

        for self.t in range(time_steps):
            self.frame = input[:,self.t,:].unsqueeze(0).to(self.dev,
                                                           torch.float)
            A = self.frame
            for i, cell in enumerate(self.cells):
                # First time step
                if self.t == 0:
                    hx = (cell.R, cell.R)
                else:
                    hx = cell.H
                    
                R, cell.H = cell.recurrent(A, hx)
                # Optional tracking
                cell.track_hidden(self.output_mode, hx)
                cell.track_representation(self.output_mode, R)
                cell.R = R

                if i < self.hparams.n_layers - 1:
                    A = cell.dense(R)
                else:
                    A_hat = cell.dense(R)
                
            pos = nn.functional.relu(A_hat - self.frame)
            neg = nn.functional.relu(self.frame - A_hat)
            E = torch.cat([pos, neg], 2)
            
            if self.output_mode == 'error':
                total_output.append(E)
            elif self.output_mode == 'prediction':
                total_output.append(A_hat)
        
        if self.output_mode == 'eval':
            return self.eval_outputs()
        else:
            return torch.stack(total_output, 2)

