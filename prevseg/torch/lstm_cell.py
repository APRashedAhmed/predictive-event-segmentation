import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V

class LSTM(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    Special args:
    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0,
                 dropout_method='pytorch', jit=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        if torch.__version__ == '1.4.0' and torch.cuda.is_available():
            self.i2h = self.i2h.cuda()
            self.h2h = self.h2h.cuda()
            
        self.reset_parameters()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(torch.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
            c_t.data.set_(torch.mul(c_t, self.mask).data)
            c_t.data *= 1.0 / (1.0 - self.dropout)

        h_t = torch.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                h_t.data.set_(torch.mul(h_t, self.mask).data)
                h_t.data *= 1.0 / (1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)

        return h_t, (h_t, c_t)
