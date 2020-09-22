"""Script for visualization routines"""
import logging

import torch
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def visualize_schapiro_walk(data, vlines=None, offset=0, title='', idx=None):
    fig = plt.figure(idx)

    len_data = len(data)
    for i, layer_data in enumerate(data):
        ax = fig.add_subplot(11 + i + len_data*100)

        if isinstance(layer_data, torch.Tensor) and layer_data.is_cuda:
            layer_data = layer_data.cpu().detach().numpy()

        ax.plot(layer_data[offset:])
        ax.set_ylabel(f'Layer {i+1}')
        if vlines is not None:
            [ax.axes.axvline(v, ls=':', label='Border Crossing')
             for v in vlines]
            if i == 0:
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())

        if i == len_data-1:
            ax.set_xlabel('Step')

    if title:
        fig.suptitle(title)

    gcf = plt.gcf()
    gcf.set_size_inches(16, 9)
    return fig

