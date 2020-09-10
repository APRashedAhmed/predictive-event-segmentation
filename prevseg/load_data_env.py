import gc
import datetime
import logging
import argparse
import socket
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint

import IPython
import pytorch_lightning as pl

from prevseg import index
from prevseg.datasets import breakfast
from prevseg.models import prednet

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dir_checkpoints', type=str,
                        default=str(index.DIR_CHECKPOINTS))
    parser.add_argument('--dir_weights', type=str,
                        default=str(index.DIR_WEIGHTS))
    parser.add_argument('--dir_logs', type=str,
                        default=str(index.DIR_LOGS_TB))
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=117)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--dataloader', type=str,
                        default='BreakfastI3DFVDataset')
    parser.add_argument('--hostname', type=str, default='')
    parser.add_argument('--train_data', type=bool, default=False)
    
    # add all the available options to the trainer
    # parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    # Check this is correct as well
    if hasattr(breakfast, hparams.dataloader):
        Dataloader = getattr(breakfast, hparams.dataloader)
    else:
        raise Exception(f'Invalid dataloader "{hparams.dataloader}" passed.')

    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f'Current time: {now}', flush=True)    

    # Get the hostname for book keeping
    hparams.hostname = hparams.hostname or socket.gethostname()

    # Set the device
    if hparams.device == 'cuda' and torch.cuda.is_available():
        print('Using GPU', flush=True)
        device = torch.device('cuda')
    else:
        print('Using CPU', flush=True)
        device = torch.device('cpu')

    # Load the testing data
    path_summary = index.DIR_BK_META / 'i3d_fvs_meta.csv'
    df_coarse = pd.read_csv(str(path_summary))
    df_counts = df_coarse.groupby(['patient', 'action']).size().reset_index(
        name='counts')
    df_p20_cam01 = df_coarse[(df_coarse.patient==20) &
                             (df_coarse.camera=='cam01')]
    ids = list(df_p20_cam01.id.values)    
    torch_tests = [] 
    for path in df_p20_cam01.path: 
        torch_tests.append(torch.from_numpy(
            np.load(path)).float().unsqueeze(0).to(device))

    # Load the segmentations
    coarse_meta = pd.read_csv(
        str(index.DIR_BK_META / 'coarse_segmentations_meta.csv'))
    coarse_meta_id = coarse_meta[coarse_meta.id.isin(ids)]
    coarse_meta_id = pd.DataFrame(df_p20_cam01.id).merge(
        coarse_meta_id, how='left', on='id')
    segmentations = []
    for path in list(coarse_meta_id.path.values):
        with open(path, 'r') as f:
            segmentations.append([line.split('-')[0] for line in f][2:-1])
            
    # Load the test data if desired
    if hparams.train_data:
        print('Loading training data', flush=True)
        ds = Dataloader()

    # Load relevant model checkpoints
    model_ckpt_stems = [ 
        'pn_3L_200E_ctrl_v1/bk_i3d_global_step=32255_epoch=191_val_loss=0.068.ckpt', 
        'pn_3L_200E_exp10_v1/bk_i3d_global_step=28559_epoch=169_val_loss=0.002.ckpt', 
        'pn_3L_200E_exp2_v1/bk_i3d_global_step=28727_epoch=170_val_loss=0.019.ckpt', 
        'pn_3L_200E_tri_v0/bk_i3d_global_step=32591_epoch=193_val_loss=0.024.ckpt',
        'prednet_6l_200e_all_llm_v0/bk_i3d_global_step=22679_epoch=134_val_loss=0.025.ckpt',
    ]
    versions = [int(path.split('/')[0][-1]) for path in model_ckpt_stems]
    model_ckpts = [str(index.DIR_CHECKPOINTS / path)
                   for path in model_ckpt_stems]
    

    # IPython.embed()

    
