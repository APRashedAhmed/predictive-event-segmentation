import datetime
import logging
import argparse
import socket
from pathlib import Path
from pprint import pprint

import IPython
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.logging.neptune import NeptuneLogger

import prevseg.constants as const
from prevseg import index, models, dataloaders

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--ipy', action='store_true')
    parser.add_argument('--model', type=str, default='PredNetTrackedSchapiro')
    parser.add_argument('--dataloader', type=str, default='ShapiroResnetEmbeddingDataset')

    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--gpus', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=117)
    parser.add_argument('--batch_size', type=int, default=256+128)
    parser.add_argument('--n_val', type=int, default=256)

    parser.add_argument('--dir_checkpoints', type=str,
                        default=str(index.DIR_CHECKPOINTS))
    parser.add_argument('--dir_weights', type=str,
                        default=str(index.DIR_WEIGHTS))
    parser.add_argument('--dir_logs', type=str,
                        default=str(index.DIR_LOGS_TB))
    parser.add_argument('--checkpoint_period', type=float, default=1.0)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--save_top_k', type=float, default=1)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--name', type=str, default='')
        
    # Get Model and Dataset specific args
    temp_args, _ = parser.parse_known_args()
    
    # Make sure this is correct
    if hasattr(models, temp_args.model):
        Model = getattr(models, temp_args.model)
        parser = Model.add_model_specific_args(parser)
    else:
        raise Exception(f"""
        Invalid model "{temp_args.model}" passed. Check it is importable:
        "from prevseg.models import {temp_args.model}"
        """)
    
    # Check this is correct as well
    if hasattr(dataloaders, temp_args.dataloader):
        Dataloader = getattr(dataloaders, temp_args.dataloader)
        parser = Dataloader.add_model_specific_args(parser)
    else:
        raise Exception(f"""
        Invalid dataloader "{temp_args.dataloader}" passed. Check it is
        importable: "from prevseg.dataloaders import {temp_args.dataloader}"
        """)
    
    # add all the available options to the trainer
    # parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    # Get or create a name
    hparams.name = Model.name if not hparams.name else hparams.name

    # Get the hostname for book keeping
    hparams.hostname = socket.gethostname()

    # Neptune Logger
    get_logger = lambda : NeptuneLogger(
        # api_key="ANONYMOUS",
        project_name="aprashedahmed/sandbox",
        experiment_name=f'{hparams.name}_{hparams.exp_name}',
        params=vars(hparams),
        tags=["pytorch-lightning", "test"]
    )

    if not hparams.load_model:
        # Checkpoint Call back
        ckpt_dir = Path(hparams.dir_checkpoints) \
            / f'{hparams.name}_{hparams.exp_name}'
        if not ckpt_dir.exists():
            ckpt_dir.mkdir(parents=True)
        ckpt = pl.callbacks.ModelCheckpoint(
            filepath=str(
                ckpt_dir /
                (hparams.exp_name+'_{global_step:05d}_{epoch:03d}_{val_loss:.3f}')),
            verbose=True,
            save_top_k=hparams.save_top_k,
            period=hparams.checkpoint_period,
        )

        # Define the trainer
        trainer = pl.Trainer(
            checkpoint_callback=ckpt,
            max_epochs=hparams.epochs,
            logger=get_logger(),
            val_check_interval=hparams.val_check_interval,
            gpus=hparams.gpus,
        )

        now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'Current time: {now}', flush=True)
        print(f'Running with following hparams:', flush=True)
        pprint(vars(hparams))

        # Define the model
        model = Model(hparams)
        print(model, flush=True)

        print('Beginning training', flush=True)
        # Train the model
        trainer.fit(model)

    else:
        # Get all the experiments with the name hparams.name*
        experiments = list(index.DIR_CHECKPOINTS.glob(f'{hparams.name}_{hparams.exp_name}*'))

        # import pdb; pdb.set_trace()
        if len(experiments) > 1:
            # Get the newest exp by v number
            experiment_newest = sorted(
                experiments, 
                key=lambda path: int(path.stem.split('_')[-1][1:]))[-1]
            # Get the model with the best (lowest) val_loss
        else:
            experiment_newest = experiments[0]
        experiment_newest_best_val = sorted(
            experiment_newest.iterdir(),
            key=lambda path: float(path.stem.split('val_loss=')[-1]))[0]
        experiment_newest_best_val
        # model, trainer = None, None
        # train_dataloader, val_dataloader = None, None
        # errors, optimizer = None, None
        # ckpt = None
        # train_errors, val_errors = None, None
        # res = None
        # gc.collect()
        # torch.cuda.empty_cache()

        model = Model.load_from_checkpoint(str(experiment_newest_best_val))
        model.prepare_data()
        model.cuda()
        model.logger = get_logger()

    # Define the visualization data
    euclid_nodes = np.array([11, 13, 14, 12, 13, 10, 12, 11, 14,
                             0,  3,  4,  2,  3,  1,  2,  0,  1,  4,
                             5,  8,  9,  7,  8,  6,  7,  5,  6,  9, 10])
    # Bidirectional euclidean walk 
    test_nodes = np.concatenate((euclid_nodes, np.flip(euclid_nodes)[1:]))
    # Where the borders are
    borders = [9, 19, 29, 30, 40, 50]

    # Create the data
    test_data = np.array([model.ds.array_data[n] for n in test_nodes]).reshape((
        1, len(test_nodes), 2048))

    print(test_data.shape,torch.Tensor(test_data).shape)

    # Visualize the test data
    figs = model.visualize(torch.Tensor(test_data), borders)

    if hparams.ipy:
        IPython.embed()    

if __name__ == '__main__':
    main()