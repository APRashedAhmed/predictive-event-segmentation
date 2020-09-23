"""Place for multilayer perceptrons"""
import logging
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.decorators import auto_move_data

import prevseg.models.prednet as pn
import prevseg.constants as const
from prevseg.models.prednet import PredCell
from prevseg.models.prednet_ablated import PredNetRelu2Tanh
from prevseg.utils import child_argparser


class DenseCell(PredCell):
    name = 'dense_cell'
    module_names = ('dense',)
    def __init__(self, parent, layer_num, hparams, a_channels, r_channels,
                 *args, **kwargs):
        self.build_recurrent = lambda *args, **kwargs : None
        self.build_update = lambda *args, **kwargs : None
        super().__init__(parent, layer_num, hparams, a_channels, r_channels,
                         *args, **kwargs)

    def build_dense(self):
        dense = nn.Sequential(nn.Linear(
            self.copies*self.a_channels[self.layer_num],
            self.r_channels[self.layer_num]))
        dense.add_module('relu', nn.ReLU())
        return dense        

    def reset(self, batch_size=None):
        batch_size = batch_size or self.hparams.batch_size
        self.E = torch.zeros(1,                  # Single time step
                             batch_size,
                             self.copies*self.a_channels[self.layer_num],
                             device=self.parent.dev)        
        self.R = torch.zeros(1,                  # Single time step
                             batch_size,
                             self.r_channels[self.layer_num],
                             device=self.parent.dev)
        self.E_loss = torch.zeros(1,                  # Single time step
                                  batch_size or self.hparams.batch_size,
                                  2*self.a_channels[self.layer_num],
                                  device=self.parent.dev)
        self.init_diff_lists()
        

class MLPErrorCode(PredNetRelu2Tanh):
    name = 'mlp_error_code'
    track = ('representation', 'error')
    def __init__(self, hparams, CellClass=DenseCell, *args, **kwargs):
        super().__init__(hparams, CellClass=CellClass, *args, **kwargs)
        # Output layer
        self.dense = nn.Sequential(
            nn.Linear(self.r_channels[hparams.n_layers - 1],
                      self.a_channels[0]),
            nn.Tanh())

    def forward(self, input, output_mode=None):
        self.output_mode = output_mode or self.output_mode
        _, time_steps, *_ = self.check_input_shape(input)
        
        self.total_error = []

        for self.t in range(time_steps):
            self.frame = input[:,self.t,:].unsqueeze(0).to(self.dev,
                                                           torch.float)
            if self.t == 0:
                self.A = torch.cat([torch.zeros_like(self.frame),
                                    torch.zeros_like(self.frame)], 2)
                
            # Loop through all the hidden layers
            for l, cell in enumerate(self.cells):
                self.A = cell.dense(self.A)
                # Optional Error tracking
                if self.t > 0:
                    cell.track_metric_diff(self.A, cell.R, 'representation')
                cell.R = self.A

            # Output Layer
            A_hat = self.dense(self.A)
                
            # Split to 2 Es
            pos = A_hat - self.frame
            neg = self.frame - A_hat
            E = F.relu(torch.cat([pos, neg], 2))
            if self.t > 0:
                cell.track_metric_diff(E, cell.E_loss, 'error')
            self.A = cell.E_loss = E

            self.track_outputs()
        
        return self.return_output()        
            
            
