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
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import prevseg.constants as const
import prevseg.dataloaders.breakfast as bk
import prevseg.dataloaders.schapiro as sch
from prevseg.torch.lstm_cell import LSTM
from prevseg.torch.activations import SatLU

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

                
class PredNet(pl.LightningModule):
    name = 'prednet'
    def __init__(self, hparams, ds=None, a_channels=None,
                 r_channels=None, CellClass=PredCell):
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
        self.a_channels = a_channels
        self.r_channels = r_channels
        self.CellClass = CellClass

        self.dev = 'cuda:0'
        
        # if self.hparams.device == 'cuda' and torch.cuda.is_available():
        #     print('Using GPU', flush=True)
        #     self.dev = 'cuda:0'
        # else:
        #     print('Using CPU', flush=True)
        #     self.dev = torch.device('cpu')

        # Put together the model
        self.build_model()                
        
    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)               

    def build_model(self):
        # Set the seed if provided
        if hasattr(self.hparams, 'seed'):
            self.set_seed(self.hparams.seed)
        
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
    
    def top_down_pass(self, t):
        # Loop backwards
        for l, cell in reversed(list(enumerate(self.cells))):
            E, R = cell.E, cell.R
            # First time step
            if t == 0:
                hx = (R, R)
            else:
                hx = cell.H

            # If not in the last layer, upsample R and
            if l < self.n_layers - 1:
                E = torch.cat((E,  cell.upsample(self.cells[l+1].R)), 2)

            cell.R, cell.H = cell.recurrent(E, hx)
            
    def bottom_up_pass(self):
        for cell in self.cells:
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
            self.A = input[:,t,:].unsqueeze(0).to(self.dev, torch.float)
            
            # Loop from top layer to update R and H
            self.top_down_pass(t)
            # Loop bottom up to get E and A
            self.bottom_up_pass()
            
            if self.output_mode == 'error':
                mean_error = torch.cat(
                    [torch.mean(cell.E.view(cell.E.size(1), -1),
                                1, keepdim=True)
                     for cell in self.cells], 1)
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
        self.ds = self.ds or bk.BreakfastI3DFVDataset()
        self.ds_length = len(self.ds)
        
        n_test, n_val = self.hparams.n_test, self.hparams.n_val
        indices = list(range(self.ds_length))
        
        self.test_sampler = SubsetRandomSampler(indices[:n_test])
        self.val_sampler = SubsetRandomSampler(indices[n_test : n_test+n_val])
        self.train_sampler = SubsetRandomSampler(indices[n_test+n_val:])
        
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

    # def test_dataloader(self):
    #     return DataLoader(self.ds, 
    #                       batch_size=self.batch_size, 
    #                       sampler=self.test_sampler,
    #                       num_workers=self.hparams.n_workers)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def _common_step(self, batch, batch_idx, mode):
        data, path = batch
        errors = self.forward(data) # batch x n_layers x nt
        loc_batch = errors.size(0)
        errors = torch.mm(errors.view(-1, self.time_steps), 
                          self.time_loss_weights) # batch*n_layers x 1
        if self.layer_loss_mode is not None:
            errors = torch.mm(errors.view(loc_batch, -1), 
                              self.layer_loss_weights)
        errors = torch.mean(errors, axis=0)
        
        if mode == 'train':
            prefix = ''
        else:
            prefix = mode + '_'
            
        self.logger.experiment.log_metric(f'{prefix}loss', 
                                          errors)
        return {f'{prefix}loss' : errors}
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    # def test_step(self, batch, batch_idx):
    #     return self._common_step(batch, batch_idx, 'test')

    def validation_epoch_end(self, output):
        out_dict = {}
        out_dict['val_loss'] = torch.as_tensor(np.mean([out['val_loss'].item()
                                                        for out in output]))
        out_dict['global_step'] = self.global_step
        return out_dict

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument('--n_layers', type=int, default=4)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--output_mode', type=str, default='error')
        parser.add_argument('--layer_loss_mode', type=str, default='first')
        return parser

    # def test_epoch_end(self, output):
    #     out_dict = {}
    #     out_dict['test_loss'] = np.mean([out['test_loss'].item()
    #                                     for out in output])
    #     out_dict['global_step'] = self.global_step
    #     return out_dict

class PredCellTracked(PredCell):
    name = 'predcell_tracked'
    """Organizational class."""
    def __init__(self, parent, layer_num, hparams, a_channels, r_channels,
                 *args, **kwargs):
        super().__init__(parent, layer_num, hparams, a_channels, r_channels,
                         *args, **kwargs)
        # Tracking
        self.hidden_full_list = []
        self.hidden_diff_list = []
        self.representation_full_list = []
        self.representation_diff_list = []
        self.error_full_list = []
        self.error_diff_list = []

    def track_representation(self, output_mode, R):
        # Track representation states if desired
        if 'representation_full' in self.parent.track and output_mode == 'eval':
            self.representation_full_list.append(R.permute(1, 0, 2))
        if 'representation_diff' in self.parent.track and output_mode == 'eval':
            diffs = torch.mean(
                (R.permute(1, 0, 2) - self.R.permute(1, 0, 2))**2,
                2).detach()
            diff_dict = {f'{self.parent.tb_labels[i]}' : diff
                         for i, diff in enumerate(diffs)}
            scalar_name = f'test_representation_diff_{self.parent.run_num}/' \
                'layer_{self.layer_num}/'
            # self.parent.logger.experiment.log_metrics(
            #     scalar_name,
            #     dict(diff_dict),
            #     self.parent.t)
            self.representation_diff_list.append(diffs)

    def track_hidden(self, output_mode, H):
        # Track hidden states if desired
        if 'hidden_full' in self.parent.track and output_mode == 'eval':
            self.hidden_full_list.append(H[1].permute(1, 0, 2))
        if 'hidden_diff' in self.parent.track and output_mode == 'eval':
            diffs = torch.mean(
                (H[1].permute(1, 0, 2) - self.H[1].permute(1, 0, 2))**2,
                2).detach()
            diff_dict = {f'{self.parent.tb_labels[i]}' : diff
                         for i, diff in enumerate(diffs)}
            scalar_name = f'test_hidden_diff_{self.parent.run_num}/' \
                'layer_{self.layer_num}/'
            # self.parent.logger.experiment.log_metrics(
            #     scalar_name,
            #     dict(diff_dict),
            #     self.parent.t)
            self.hidden_diff_list.append(diffs)

    def track_error(self, output_mode, E):
        # Track hidden states if desired
        if 'error_full' in self.parent.track and output_mode == 'eval':
            self.error_full_list.append(E.permute(1, 0, 2))
        if 'error_diff' in self.parent.track and output_mode == 'eval':
            # print('E', E.shape, self.E.shape)
            diffs = torch.mean(
                (E.permute(1, 0, 2) - self.E.permute(1, 0, 2))**2,
                2).detach()
            diff_dict = {f'{self.parent.tb_labels[i]}' : diff
                         for i, diff in enumerate(diffs)}
            scalar_name = f'test_error_diff_{self.parent.run_num}/' \
                'layer_{self.layer_num}/'
            # self.parent.logger.experiment.add_scalars(
            #     scalar_name,
            #     dict(diff_dict),
            #     self.parent.t)
            # self.parent.logger.experiment.add_scalars(
            #     f'test_error_diff_{self.parent.run_num}/layer_{self.layer_num}/',
            #     {f'{self.parent.tb_labels[i]}' for i, diff in enumerate(diffs)},
            #     self.parent.t)
            self.error_diff_list.append(diffs)

    def reset(self, *args, **kwargs):
        self.hidden_full_list = []
        self.hidden_diff_list = []
        self.representation_full_list = []
        self.representation_diff_list = []        
        self.error_full_list = []
        self.error_diff_list = []
        return super().reset(*args, **kwargs)
        
            
class PredNetTracked(PredNet):
    name = 'prednet_tracked'
    def __init__(self, hparams, track=None, CellClass=PredCellTracked, *args,
                 **kwargs):
        super().__init__(hparams, CellClass=CellClass, *args, **kwargs)
        self.track = track or ['hidden_diff',
                               'error_diff',
                               'representation_diff']
        self.run_num = None
        self.tb_labels = None
        
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
            cell.track_representation(self.output_mode, R)
            cell.track_hidden(self.output_mode, H)

            # Update cell state
            cell.R, cell.H = R, H
            
    def bottom_up_pass(self):
        for cell in self.cells:
            # Go from R to A_hat
            A_hat = cell.dense(cell.R)

            # Convenience
            if self.output_mode == 'prediction' and cell.layer_num == 0:
                self.frame_prediction = A_hat

            # Split to 2 Es
            pos = F.relu(A_hat - self.A)
            neg = F.relu(self.A - A_hat)
            E = torch.cat([pos, neg], 2)
            
            # Optional Error tracking
            cell.track_error(self.output_mode, E)

            # Update cell error
            cell.E = E

            # If not last layer, update stored A
            if cell.layer_num < self.n_layers - 1:
                self.A = cell.update_a(E)
            
    def forward(self, input, output_mode=None, track=None, run_num=None,
                tb_labels=None):
        self.run_num = run_num or self.run_num
        self.tb_labels = tb_labels or self.tb_labels
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
            # track_list = getattr(self.cells[0], tracked+'_list')
            # [print(t.shape) for t in track_list]
            
            outputs[tracked] = [torch.cat(getattr(cell, tracked+'_list'), 1)
                                for cell in self.cells]
        return outputs

    def _visualize(self, data, vlines=None, offset=0):
        fig = plt.figure()
        ax_large = fig.add_subplot(111)

        len_data = len(data)
        for i, layer_data in enumerate(data):
            ax = fig.add_subplot(11 + i + len_data*100)[offset:]
            ax.plot(layer_data)
            ax.set_ylabel(f'Layer {i+1}')
            if vlines is not None:
                [ax.axes.axvline(v, ls=':') for v in vlines]
            if i == len_data-1:
                ax.set_xlabel('Step')

        ax_large.axes.xaxis.set_ticks([])
        ax_large.axes.yaxis.set_ticks([])
        gcf = plt.gcf()
        gcf.set_size_inches(16, 9)
        return fig

    def visualize_errors(self, test_data, borders=None):
        outs_pe = self.forward(test_data,
                               output_mode='error',
                               run_num='fwd_rev', 
                               tb_labels=['nodes'])
        outs_array = outs_pe[0,:,:].cpu().detach().numpy()
        return self._visualize(outs_array, vlines=borders)
        
    def visualize_diffs(self, test_data, borders=None):
        outs = self.forward(test_data,
                            output_mode='eval',
                            run_num='fwd_rev', 
                            tb_labels=['nodes'])
        figs = []
        for diff in self.track:
            offset = 0 if 'error' in track else 1
            figs.append(self._visualize(test_data[diff],
                                  vlines=borders,
                                  offset=offset))
        return figs
    
    def visualize(self, *args, **kwargs):
        return self.visualize_diffs(*args, **kwargs) + \
            [self.visualize_errors(*args, **kwargs)]

class PredNetTrackedSchapiro(PredNetTracked):
    name = 'prednet_tracked_schapiro'
    def prepare_data(self):
        self.ds = self.ds or sch.ShapiroResnetEmbeddingDataset(
            batch_size=self.batch_size, 
            n_pentagons=self.hparams.n_pentagons, 
            max_steps=self.hparams.max_steps, 
            n_paths=self.hparams.n_paths)
        self.ds_val = sch.ShapiroResnetEmbeddingDataset(
            batch_size=self.batch_size, 
            n_pentagons=self.hparams.n_pentagons,
            n_paths=1,
            mapping=self.ds.mapping,
            mode='euclidean')
                
    def train_dataloader(self):
        return DataLoader(self.ds, 
                          batch_size=None,
                          num_workers=self.hparams.n_workers)
    
    def val_dataloader(self):
        return DataLoader(self.ds_val, 
                          batch_size=None,
                          num_workers=self.hparams.n_workers)
