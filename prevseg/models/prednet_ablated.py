"""Prednet with portions removed for comparison."""
import logging

import torch
import torch.nn.functional as F

from prevseg.models import prednet as pn

logger = logging.getLogger(__name__)

class PredCellRelu2Tanh(pn.PredCellTracked):
    def reset(self, batch_size=None):
        self.E_loss = torch.zeros(1,                  # Single time step
                                  batch_size or self.hparams.batch_size,
                                  2*self.a_channels[self.layer_num],
                                  device=self.parent.dev)
        return super().reset(batch_size=batch_size)

    
class PredNetRelu2Tanh(pn.PredNetTrackedSchapiro):
    """PredNet unexpectedly recapitulates human fmri data. Perhaps it is related
    to the error code being positive via the relus. See what happens to the
    representations when coding a signed error coding scheme.
    """
    name = 'prednet_relu2tanh'
    track = ('representation', 'hidden', 'error_relu', 'error_tanh')
    def __init__(self, hparams, CellClass=PredCellRelu2Tanh, *args, **kwargs):
        super().__init__(hparams, CellClass=CellClass, *args, **kwargs)
    
    def bottom_up_pass(self):
        for cell in self.cells:
            # Go from R to A_hat
            A_hat = cell.dense(cell.R)

            # Convenience
            if self.output_mode == 'prediction' and cell.layer_num == 0:
                self.frame_prediction = A_hat

            # # Split to 2 Es
            pos = A_hat - self.A
            neg = self.A - A_hat
            
            E_relu = F.relu(torch.cat([pos, neg], 2))
            cell.track_metric_diff(E_relu, cell.E_loss, 'error_relu')
            # Update the loss error
            cell.E_loss = E_relu

            # Keep above to use as the loss, create tanh for signalling
            E_tanh = F.tanh(torch.cat([pos, neg], 2))
            cell.track_metric_diff(E_tanh, cell.E, 'error_tanh')
            
            # Update cell error
            cell.E = E_tanh

            # If not last layer, update stored A
            if cell.layer_num < self.n_layers - 1:
                self.A = cell.update_a(E_tanh)

    def track_outputs(self):
        if self.output_mode == 'error':
            mean_error = torch.cat(
                [torch.mean(cell.E_loss.view(cell.E_loss.size(1), -1),
                            1, keepdim=True)
                 for cell in self.cells], 1)
            # batch x n_layers
            self.total_error.append(mean_error)
                

class PredNetAblatedError(pn.PredNetTrackedSchapiro):
    """
    This doesnt seem to work for some reason. Need to think more carefully
    about it.
    """
    name = 'prednet_error_ablated'
    track = ('representation_diff', 'hidden_diff')
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)
    
    def bottom_up_pass(self):
        for cell in self.cells:
                
            # Go from R to A_hat
            A_hat = cell.dense(cell.R)

            # Convenience            
            if cell.layer_num == 0:
                self.frame = self.A
            if self.output_mode == 'error' and cell.layer_num == 0:
                pos = F.relu(A_hat - self.frame)
                neg = F.relu(self.frame - A_hat)
                self.frame_error = torch.cat([pos, neg], )

            # Remove these
            # # Split to 2 Es
            # pos = F.relu(A_hat - self.A)
            # neg = F.relu(self.A - A_hat)
            # E = torch.cat([pos, neg], 2)

            # E Now just has two copies of A_hat
            E = torch.cat([A_hat, A_hat], 2)

            # No Need to track this
            # # Optional Error tracking
            # cell.track_error(self.output_mode, E)

            # Update cell error
            cell.E = E

            # If not last layer, update stored A
            if cell.layer_num < self.n_layers - 1:
                self.A = cell.update_a(E)

    def track_outputs(self):
        if self.output_mode == 'error':
            self.total_error.append(torch.cat(
                [torch.mean(self.frame_error.view(self.frame_error.size(1), -1),
                            1, keepdim=True)] * self.hparams.n_layers, 1))
                                    

        

