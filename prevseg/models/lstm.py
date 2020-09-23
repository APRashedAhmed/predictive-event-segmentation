"""Place for 'vanilla' LSTM models"""
import logging
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.decorators import auto_move_data

import prevseg.models.prednet as pn
import prevseg.constants as const
from prevseg.utils import child_argparser

logger = logging.getLogger(__name__)

class LSTMCell(pn.PredCell):
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


class LSTMCellDense(LSTMCell, pn.PredCell):
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


class LSTMStacked(pn.PredNet):
    name = 'lstmstacked'
    track = ('representation', 'hidden')
    def __init__(self, hparams, CellClass=LSTMCell, a_channels=None,
                 r_channels=None, *args, **kwargs):
        if not isinstance(hparams, Namespace):
            hparams = Namespace(**hparams)
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
        
    def build_dense(self, r_channels, a_channels):
        return nn.Sequential(nn.Linear(r_channels, a_channels), nn.Tanh())

    @auto_move_data
    def forward(self, input, output_mode=None):
        self.output_mode = output_mode or self.output_mode
        _, time_steps, *_ = self.check_input_shape(input)
        
        total_output = []

        for self.t in range(time_steps-1):
            self.input_frame = input[:,self.t,:].unsqueeze(0).to(self.dev,
                                                                 torch.float)
            self.target_frame = input[:,self.t+1,:].unsqueeze(0).to(self.dev,
                                                                    torch.float)
            self.A = self.input_frame
            self._cell_ops()
            
            pos = F.relu(self.A_hat - self.target_frame)
            neg = F.relu(self.target_frame - self.A_hat)
            self.E = torch.cat([pos, neg], 2)
            
            if self.output_mode == 'error':
                total_output.append(self.E)
            elif self.output_mode == 'prediction':
                total_output.append(self.A_hat)
        
        if self.output_mode == 'eval':
            return self.eval_outputs()
        else:
            return torch.stack(total_output, 2)

    def _cell_ops(self):
        for i, cell in enumerate(self.cells):
            # First time step
            if self.t == 0:
                hx = (cell.R, cell.R)
            else:
                hx = cell.H

            R, H = cell.recurrent(self.A, hx)
            
            # Optional tracking
            cell.track_metric_diff(R, cell.R, 'representation')
            cell.track_metric_diff(H[1], cell.H[1], 'hidden')

            cell.R, cell.H = R, H
                
            if i < self.hparams.n_layers - 1:
                self.A = cell.R
            else:
                self.A_hat = self.dense(cell.R)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = child_argparser(
            pn.PredNet.add_model_specific_args(parent_parser))
        parser.add_argument('--layer_loss_mode', type=str, default='')

        # See if we have the right number of inputs and n_layers has been
        # specified to infer default batch size
        temp_args, _ = parser.parse_known_args()
        if temp_args.input_size is not None and temp_args.input_size <= 2048:
            if temp_args.n_layers is not None:
                default_batch_size = 256
                if temp_args.n_layers == 2:
                    default_batch_size = 256 + 128
                elif temp_args.n_layers == 1:
                    default_batch_size = 512 + 32
                parser.add_argument('-b', '--batch_size', type=int,
                                    default=default_batch_size)
        return parser

    def _common_step(self, batch, batch_idx):
        data, path = batch
        prediction_errors = self.forward(data) # batch x n_layers x nt
        loc_batch = prediction_errors.size(0)
        loss = torch.mm(prediction_errors.view(-1, self.time_steps-1), 
                          self.time_loss_weights) # batch*n_layers x 1
        if self.layer_loss_mode is not None:
            loss = torch.mm(loss.view(loc_batch, -1), 
                              self.layer_loss_weights)
        return torch.mean(loss, axis=0)

    def build_time_loss_weights(self, time_steps=None):
        time_steps = time_steps or self.time_steps
        # How much to weight errors at each timestep
        time_loss_weights = 1. / (time_steps-2) * torch.ones(time_steps-1, 1,
                                                             device=self.dev)
        # Dont count first time step
        time_loss_weights[0] = 0
        return time_loss_weights
        

class LSTMStackedDense(LSTMStacked):
    name = 'lstmstacked_dense'
    def __init__(self, hparams, CellClass=LSTMCellDense, a_channels=None,
                 r_channels=None, *args, **kwargs):
        self.build_dense = lambda *args, **kwargs : None
        # Run the init and cleanup
        super().__init__(hparams=hparams, CellClass=CellClass,
                         r_channels=r_channels, a_channels=a_channels, *args,
                         **kwargs)        

    def _cell_ops(self):
        for i, cell in enumerate(self.cells):
            # First time step
            if self.t == 0:
                hx = (cell.R, cell.R)
            else:
                hx = cell.H

            R, H = cell.recurrent(self.A, hx)
            
            # Optional tracking
            cell.track_metric_diff(R, cell.R, 'representation')
            cell.track_metric_diff(H[1], cell.H[1], 'hidden')

            cell.R, cell.H = R, H
                        
            if i < self.hparams.n_layers - 1:
                self.A = cell.dense(cell.R)
            else:
                self.A_hat = cell.dense(cell.R)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = child_argparser(LSTMStacked.add_model_specific_args(
            parent_parser))
        temp_args, _ = parser.parse_known_args()
        if temp_args.input_size is not None and temp_args.input_size <= 2048:
            if temp_args.n_layers is not None:
                default_batch_size = 256
                if temp_args.n_layers == 2:
                    default_batch_size = 256 + 64 + 32
                elif temp_args.n_layers == 1:
                    default_batch_size = 512 + 32
                parser.add_argument('-b', '--batch_size', type=int,
                                    default=default_batch_size)
        return parser
                
