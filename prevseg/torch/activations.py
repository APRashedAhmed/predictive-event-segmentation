"""Generic activation functions"""
import logging

import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class SatLU(nn.Module):

    def __init__(self, lower=0, upper=255, inplace=False):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' ('\
            + 'min_val=' + str(self.lower) \
	        + ', max_val=' + str(self.upper) \
	        + inplace_str + ')'
