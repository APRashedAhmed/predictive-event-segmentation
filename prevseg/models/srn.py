"""Place for simple recurrent models."""
import logging
from argparse import Namespace

import torch
import torch.nn as nn
from pytorch_lightning.core.decorators import auto_move_data

import prevseg.models.prednet as pn
import prevseg.constants as const
from prevseg.models.lstm import LSTMCellDense, LSTMStackedDense
from prevseg.utils import child_argparser

class SRNCell(LSTMCellDense):
    name = 'srncell'
    def __init__(self, parent, layer_num, hparams, a_channels, r_channels,
                 *args, **kwargs):
        self.build_update = lambda *args, **kwargs : None
        super().__init__(parent, layer_num, hparams, a_channels, r_channels,
                         RecurrentClass=nn.RNN, *args, **kwargs)
        
    def build_recurrent(self):
        recurrent = self.RecurrentClass(
            self.a_channels[self.layer_num],
            self.r_channels[self.layer_num],
            nonlinearity='relu',
            # batch_first=True,
        )
        recurrent.reset_parameters()
        return recurrent

    def reset(self, batch_size=None):
        batch_size = batch_size or self.hparams.batch_size
        self.R = torch.zeros(1,                  # Single time step
                             batch_size,
                             self.r_channels[self.layer_num],
                             device=self.parent.dev)
        self.H = torch.zeros(1,                  # Single time step
                              batch_size,
                              self.r_channels[self.layer_num],
                              device=self.parent.dev)
        self.hidden_full_list = []
        self.hidden_diff_list = []
        self.representation_full_list = []
        self.representation_diff_list = []

    def track_hidden(self, output_mode, H):
        # Track hidden states if desired
        if 'hidden_full' in self.parent.track and output_mode == 'eval':
            self.hidden_full_list.append(H.permute(1, 0, 2))
        if 'hidden_diff' in self.parent.track and output_mode == 'eval':
            diff = torch.mean(
                (H.permute(1, 0, 2) - self.H.permute(1, 0, 2))**2).detach()
            scalar_name = f'hidden_diff_layer_{self.layer_num}'
            self.parent.logger.experiment.log_metric(scalar_name, diff)
            self.hidden_diff_list.append(diff)
        

class SRN(LSTMStackedDense):
    def __init__(self, hparams, CellClass=SRNCell, *args, **kwargs):
        super().__init__(hparams=hparams, CellClass=CellClass, *args, **kwargs)

    def _cell_ops(self):
        for i, cell in enumerate(self.cells):
            h = cell.H
            cell.R, cell.H = cell.recurrent(self.A, h)
            # Optional tracking
            cell.track_hidden(self.output_mode, h)
            cell.track_representation(self.output_mode, self.A)
            
            if i < self.hparams.n_layers - 1:
                self.A = cell.dense(cell.R)
            else:
                self.A_hat = cell.dense(cell.R)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = child_argparser(LSTMStackedDense.add_model_specific_args(
            parent_parser))
        parser.add_argument('--n_layers', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=512+128+64+32)
        return parser
        
