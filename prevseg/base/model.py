"""BaseModel to be used"""
import logging

import torch
from omegaconf import OmegaConf, DictConfig

from prevseg.utils import child_argparser

logger = logging.getLogger(__name__)


class BaseTorchModel:
    name = 'base_torch_model'
    def __init__(self, hparams, *args, **kwargs):
        """Ensure hparams is a OmegaConf"""
        super().__init__(*args, **kwargs)
        if isinstance(hparams, DictConfig):
            self.hparams = hparams
        elif isinstance(hparams, dict):
            self.hparams = OmegaConf.create(hparams)
        elif hasattr(hparams, '__dict__'):
            self.hparams = OmegaConf.create(vars(hparams))
        else:
            self.hparams = hparams

        self.n_layers = self.hparams.n_layers
        self.input_size = self.hparams.input_size
        self.time_steps = self.hparams.time_steps
        self.batch_size = self.hparams.batch_size
        self.lr = self.hparams.lr or self.hparams.learning_rate
            
    def build_time_loss_weights(self, time_steps=None):
        time_steps = time_steps or self.time_steps
        # How much to weight errors at each timestep
        time_loss_weights = 1. / (time_steps-1) * torch.ones(time_steps, 1,
                                                             device=self.dev)
        # Dont count first time step
        time_loss_weights[0] = 0
        return time_loss_weights
                
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = child_argparser(parent_parser)
        parser.add_argument('--n_layers', type=int, default=1)
        parser.add_argument('--lr', type=float, default=0.001)
        return parser
    
