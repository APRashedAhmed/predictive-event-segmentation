"""Place for simple recurrent models."""
import logging
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.decorators import auto_move_data

import prevseg.models.prednet as pn
import prevseg.constants as const
from prevseg.models.lstm import LSTMCellDense, LSTMStackedDense
from prevseg.utils import child_argparser


class SRNModule(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity='tanh',
                         bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = F.tanh if nonlinearity == 'tanh' else F.relu
        self.bias = bias

        self.linear = nn.Linear(self.input_size+self.hidden_size,
                                self.input_size+self.hidden_size,
                                bias=self.bias)

    def forward(self, x, h, cat_dim=-1):
        # Assumes last dim is cat dim
        x = x.view(x.shape[1], -1)
        h = h.view(h.shape[1], -1)
        
        xh0 = torch.cat((x, h), cat_dim)
        yh1 = self.nonlinearity(self.linear(xh0))
        
        y, h1 = yh1[:, :self.input_size], yh1[:, self.input_size:]

        y = y.view(1, y.shape[0], -1)
        h1 = h1.view(1, h1.shape[0], -1)
        
        return y, h1

    
class SRNCell(LSTMCellDense):
    name = 'srncell'
    def __init__(self, parent, layer_num, hparams, a_channels, r_channels,
                 *args, **kwargs):
        self.build_update = lambda *args, **kwargs : None
        super().__init__(parent, layer_num, hparams, a_channels, r_channels,
                         RecurrentClass=SRNModule, *args, **kwargs)
        
    def build_recurrent(self):
        recurrent = self.RecurrentClass(
            self.a_channels[self.layer_num],
            self.r_channels[self.layer_num],
            nonlinearity='relu',
            # batch_first=True,
        )
        # recurrent.reset_parameters()
        return recurrent

    def reset(self, batch_size=None):
        batch_size = batch_size or self.hparams.batch_size
        self.R = torch.randn(1,                  # Single time step
                             batch_size,
                             self.r_channels[self.layer_num],
                             device=self.parent.dev)
        self.H = torch.randn(1,                  # Single time step
                             batch_size,
                             self.r_channels[self.layer_num],
                             device=self.parent.dev)
        self.hidden_diff_list = []
        self.representation_diff_list = []
        

class SRN(LSTMStackedDense):
    name = 'srn'
    def __init__(self, hparams, CellClass=SRNCell, *args, **kwargs):
        super().__init__(hparams=hparams, CellClass=CellClass, *args, **kwargs)

    def _cell_ops(self):
        for i, cell in enumerate(self.cells):
            R, H = cell.recurrent(self.A, cell.H)
            
            # Optional tracking
            cell.track_metric_diff(R, cell.R, 'representation')
            cell.track_metric_diff(H, cell.H, 'hidden')

            # Update the cells
            cell.R, cell.H = R, H
            
            # Set according to layer
            if i < self.hparams.n_layers - 1:
                self.A = cell.dense(cell.R)
            else:
                self.A_hat = cell.dense(cell.R)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = child_argparser(LSTMStackedDense.add_model_specific_args(
            parent_parser))
        parser.add_argument('--n_layers', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=512+128)
        return parser
