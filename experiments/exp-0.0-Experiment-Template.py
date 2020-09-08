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

import prevseg.models.prednet as pn
import prevseg.dataloaders.schapiro as sch
import prevseg.constants as const
from prevseg import index

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--time_steps', type=int, default=128)
    parser.add_argument('--max_steps', type=int, default=128)
    parser.add_argument('--n_paths', type=int, default=16)
    parser.add_argument('--n_pentagons', type=int, default=3)
    parser.add_argument('--dir_checkpoints', type=str,
                        default=str(index.DIR_CHECKPOINTS))
    parser.add_argument('--dir_weights', type=str,
                        default=str(index.DIR_WEIGHTS))
    parser.add_argument('--dir_logs', type=str,
                        default=str(index.DIR_LOGS_TB))
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output_mode', type=str, default='error')
    parser.add_argument('--n_val', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=117)
    parser.add_argument('--batch_size', type=int, default=256+128)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='PredNetTrackedSchapiro')
    parser.add_argument('--dataloader', type=str, default='ShapiroResnetEmbeddingDataset')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--checkpoint_period', type=float, default=1.0)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--save_top_k', type=float, default=1)
    parser.add_argument('--gpus', type=float, default=1)
    parser.add_argument('--layer_loss_mode', type=str, default='first')
    parser.add_argument('--mini', type=bool, default=False)
    parser.add_argument('--hostname', type=str, default='')
    parser.add_argument('--ipython', type=bool, default=False)
    parser.add_argument('--train_model', type=bool, default=True)
    
    # add all the available options to the trainer
    # parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    # Make sure this is correct
    if hasattr(pn, hparams.model):
        Model = getattr(pn, hparams.model)
        hparams.name = Model.name if not hparams.name else hparams.name
    else:
        raise Exception(f'Invalid model "{hparams.model}" passed.')

    # Check this is correct as well
    if hasattr(sch, hparams.dataloader):
        Dataloader = getattr(sch, hparams.dataloader)
    else:
        raise Exception(f'Invalid dataloader "{hparams.dataloader}" passed.')

    # Get the hostname for book keeping
    hparams.hostname = hparams.hostname or socket.gethostname()

    # Neptune Logger
    logger = NeptuneLogger(
        # api_key="ANONYMOUS",
        project_name="aprashedahmed/sandbox",
        experiment_name=f'{hparams.name}_{hparams.exp_name}',
        params=vars(hparams),
        tags=["pytorch-lightning", "test"]
    )
    # # Tensorboard logger
    # log_dir = Path(hparams.dir_logs) / hparams.exp_name / f'{hparams.name}'
    # if not log_dir.exists():
    #     log_dir.mkdir(parents=True)
    # logger = pl.loggers.TensorBoardLogger(str(log_dir.parent),
    #                                       name=hparams.name)

    if hparams.train_model:
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
            logger=logger,
            val_check_interval=hparams.val_check_interval,
            gpus=hparams.gpus,
        )

        now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'Current time: {now}')
        print(f'Running with following hparams:')
        pprint(vars(hparams))

        # Define the model
        model = Model(hparams=hparams)
        print(model, flush=True)

        print('Beginning training')
        # Train the model

        trainer.fit(model)

    else:
        # Get all the experiments with the name hparams.name*
        experiments = index.DIR_CHECKPOINTS.glob(f'{hparams.name}*')
        # Get the newest exp by v number
        experiment_newest = sorted(
            experiments, 
            key=lambda path: int(path.stem.split('_')[-1][1:]))[-1]
        # Get the model with the best (lowest) val_loss
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

        model = ModelClass.load_from_checkpoint(str(experiment_newest_best_val))
        model.prepare_data()
        model.cuda()
        hparams = model.hparams
        model.logger = logger

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

    # Visualize the test data
    figs = model.visualize(torch.Tensor(test_data), borders)

    if hparams.ipython:
        IPython.embed()
