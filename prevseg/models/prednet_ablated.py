"""Prednet with portions removed for comparison."""
import logging

import torch
import torch.nn.functional as F

from prevseg.models import prednet as pn

logger = logging.getLogger(__name__)


class PredNetReLU2Tanh(pn.PredNetTrackedSchapiro):
    name = 'prednet_relu2tanh'
    def bottom_up_pass(self):
        for cell in self.cells:
            # Go from R to A_hat
            A_hat = cell.dense(cell.R)

            # Convenience
            if self.output_mode == 'prediction' and cell.layer_num == 0:
                self.frame_prediction = A_hat

            # # Split to 2 Es
            pos = F.relu(A_hat - self.A)
            neg = F.relu(self.A - A_hat)
            E_relu = torch.cat([pos, neg], 2)

            # Optional Error tracking
            cell.track_error(self.output_mode, E)

            # Keep above to use as the loss, create tanh for signalling
            pos = F.tanh(A_hat - self.A)
            neg = F.tanh(self.A - A_hat)       
            E_tanh = torch.cat([pos, neg], 2)
            
            # Update cell error
            cell.E = E

            # If not last layer, update stored A
            if cell.layer_num < self.n_layers - 1:
                self.A = cell.update_a(E)

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
                

class PredNetAblatedError(pn.PredNetTrackedSchapiro):
    """
    This doesnt seem to work for some reason. Need to think more carefully
    about it.
    """
    name = 'prednet_error_ablated'
    track = ('representation_diff', 'hidden_diff')
    def __init__(self, hparams, *args, **kwargs):
        assert hparams.layer_loss_mode == 'first'
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
                                    

        

