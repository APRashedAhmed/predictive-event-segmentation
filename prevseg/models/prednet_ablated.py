"""Prednet with portions removed for comparison."""
import logging

import torch
import torch.nn.functional as F

from prevseg.utils import child_argparser
from prevseg.models import prednet as pn

logger = logging.getLogger(__name__)


class PredCellELoss(pn.PredCell):
    def reset(self, batch_size=None):
        self.E_loss = torch.zeros(1,                  # Single time step
                                  batch_size or self.hparams.batch_size,
                                  2*self.a_channels[self.layer_num],
                                  device=self.parent.dev)
        return super().reset(batch_size=batch_size)
    
    
class PredNetRelu2Tanh(pn.PredNet):
    """PredNet unexpectedly recapitulates human fmri data. Perhaps it is related
    to the error code being positive via the relus. See what happens to the
    representations when coding a signed error coding scheme.

    Result: Structure learning is preserved, if not accentuated.
    """
    name = 'prednet_relu2tanh'
    track = ('representation', 'hidden', 'error_relu', 'error_tanh')
    def __init__(self, hparams, CellClass=PredCellELoss, *args, **kwargs):
        super().__init__(hparams, CellClass=CellClass, *args, **kwargs)
    
    def bottom_up_pass(self):
        for l, cell in enumerate(self.cells):
            # Go from R to A_hat
            A_hat = cell.dense(cell.R)

            # Convenience
            if self.output_mode == 'prediction' and cell.layer_num == 0:
                self.frame_prediction = A_hat

            # Split to 2 Es
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
            if l < self.n_layers - 1:
                self.A = cell.update_a(E_tanh)

    def track_outputs(self):
        if self.output_mode == 'error':
            mean_error = torch.cat(
                [torch.mean(cell.E_loss.view(cell.E_loss.size(1), -1),
                            1, keepdim=True)
                 for cell in self.cells], 1)
            # batch x n_layers
            self.total_error.append(mean_error)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = child_argparser(
            pn.PredNet.add_model_specific_args(parent_parser))

        # See if n_layers has been specified to infer default batch size
        temp_args, _ = parser.parse_known_args()
        if temp_args.n_layers is not None:
            if temp_args.n_layers == 2:
                default_batch_size = 256 + 64 + 32
            elif temp_args.n_layers == 1:
                default_batch_size = 512 + 128 + 64
            else: 
                default_batch_size = 256
                
        parser.add_argument('-b', '--batch_size', type=int,
                            default=default_batch_size)
        return parser

    
class PredNetErrorAblated(PredNetRelu2Tanh):
    """It's seeming like the error code is the reason for the representations.
    Let's try removing it entirely, in favor of a classical FF drive. This
    *should* recreate the LSTMStacked curves.

    Results 1: Doesn't seem to reproduce, but doesnt learn structure. Seems like
    the only difference is the size of the input. Try reducing to the
    LSTMStacked size, and see again.

    
    """
    name = 'prednet_error_ablated'
    track = ('representation', 'hidden', 'error')
    def bottom_up_pass(self):
        for l, cell in enumerate(self.cells):
            # Go from R to A_hat
            A_hat = cell.dense(cell.R)

            # Convenience
            if self.output_mode == 'prediction' and cell.layer_num == 0:
                self.frame_prediction = A_hat
            
            # Split to 2 Es
            pos = A_hat - self.A
            neg = self.A - A_hat
            
            E = F.relu(torch.cat([pos, neg], 2))
            cell.track_metric_diff(E, cell.E_loss, 'error')
            # Update the loss error
            cell.E_loss = E
            
            # Update cell error to be the activity inputted
            cell.E = torch.cat([self.A, self.A])

            # If not last layer, update stored A for the next layer
            if l < self.n_layers - 1:
                self.A = cell.update_a(cell.E)  


class PredNetRecurrenceAblated(pn.PredNet):
    """It could potentially be a combination of the error code and the
    recurrence. Try ablating that to see what happens.
    """
    name = 'prednet_recurrence_ablated'
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

            # # Update cell state
            # cell.R, cell.H = R, H

            # Only update the representations
            cell.R = R
