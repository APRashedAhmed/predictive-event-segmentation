import argparse
import datetime
import logging
import socket
import time
from pathlib import Path
from pprint import pprint

import ipdb
import IPython
import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers.neptune import NeptuneLogger

import prevseg.constants as const
from prevseg import index, models, datasets
from prevseg.utils import name_from_hparams

logger = logging.getLogger(__name__)

def main(parser):
    parser.add_argument('-m', '--model', type=str,
                        default='PredNet')
    parser.add_argument('-d', '--dataset', type=str,
                        default='SchapiroResnetEmbeddingDataset')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--ipy', action='store_true')
    parser.add_argument('--no_graphs', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--user', type=str, default='aprashedahmed')
    parser.add_argument('--project', type=str, default='sandbox')
    parser.add_argument('--tags', nargs='+')
    parser.add_argument('--no_checkpoints', action='store_true')
    parser.add_argument('--offline_mode', action='store_true')
    
    parser.add_argument('--test_checkpoints', action='store_true')
    parser.add_argument('--test_epochs', type=int, default=2)
    parser.add_argument('--test_n_paths', type=int, default=2)
    parser.add_argument('--test_online', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=25)
    parser.add_argument('--gpus', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-s', '--seed', type=str, default='random')
    parser.add_argument('-b', '--batch_size', type=int, default=256+128)
    parser.add_argument('--n_val', type=int, default=1)
    parser.add_argument('--mapping', type=str, default='random')

    parser.add_argument('--dir_checkpoints', type=str,
                        default=str(index.DIR_CHECKPOINTS))
    parser.add_argument('--checkpoint_period', type=float, default=1.0)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--save_top_k', type=float, default=1)
    parser.add_argument('--early_stop_mode', type=str, default='auto')
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--early_stop_min_delta', type=str, default='0.000')

    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--exp_prefix', type=str, default='')
    parser.add_argument('--exp_suffix', type=str, default='')
    

    # Get Model and Dataset specific args
    temp_args, _ = parser.parse_known_args()

    # Make sure this is correct
    if hasattr(datasets, temp_args.dataset):
        Dataset = getattr(datasets, temp_args.dataset)
        parser = Dataset.add_dataset_specific_args(parser)
    else:
        raise Exception(
            f'Invalid dataset "{temp_args.dataset}" passed. Check it is '
            f'importable: "from prevseg.datasets import {temp_args.dataset}"'
        )

    # Get temp args now with dataset args added
    temp_args, _ = parser.parse_known_args()
    
    # Check this is correct as well
    if hasattr(models, temp_args.model):
        Model = getattr(models, temp_args.model)
        parser = Model.add_model_specific_args(parser)
    else:
        raise Exception(
            f'Invalid model "{temp_args.model}" passed. Check it is importable:'
            f' "from prevseg.models import {temp_args.model}"'
        )
        
    # Get the parser and turn into an omegaconf
    hparams = OmegaConf.create(vars(parser.parse_args()))

    # If we are test-running, do a few things differently (scale down dataset,
    # send to sandbox project, etc.)
    if hparams.test_run:
        hparams.epochs = hparams.test_epochs
        hparams.n_paths = hparams.test_n_paths
        hparams.name = '_'.join(filter(None, ['test_run', hparams.exp_prefix]))
        hparams.project = 'sandbox'
        hparams.verbose = True
        hparams.ipdb = True
        hparams.no_checkpoints = not hparams.test_checkpoints
        hparams.offline_mode = not hparams.test_online

    # Seed is a string to allow for None/random as an input. Make it passable
    # to pl.seed_everything
    hparams.seed = None if 'None' in hparams.seed or hparams.seed == 'random' \
        else int(hparams.seed)
    
    # Get the hostname for book keeping
    hparams.hostname = socket.gethostname()
    
    # Set the seed
    hparams.seed = pl.seed_everything(hparams.seed)

    # Turn the string entry for mapping into a dict (that is also a str)
    if hparams.mapping == 'default':
        hparams.mapping = const.DEFAULT_MAPPING
    elif hparams.mapping == 'random':
        hparams.mapping = str(Dataset.random_mapping(
            n_pentagons=hparams.n_pentagons))
    else:
        raise ValueError(f'Invalid entry for mapping: {hparams.mapping}')

    # Set the validation path
    hparams.val_path = str(const.DEFAULT_PATH)

    # Create experiment name
    hparams.name = name_from_hparams(hparams)
    hparams.exp_name = name_from_hparams(hparams, short=True)
    if hparams.verbose:
        print(f'Beginning experiment: "{hparams.name}"')    
    
    # Neptune Logger
    logger = NeptuneLogger(
        project_name=f"{hparams.user}/{hparams.project}",
        experiment_name=hparams.exp_name,
        params=vars(hparams),
        tags=hparams.tags,
        offline_mode=hparams.offline_mode,
    )
    
    if not hparams.load_model:
        # Checkpoint Call back
        if hparams.no_checkpoints:
            checkpoint = False
        else:
            dir_checkpoints_experiment = (Path(hparams.dir_checkpoints) / 
                                          hparams.name)
            if not dir_checkpoints_experiment.exists():
                dir_checkpoints_experiment.mkdir(parents=True)
            
            checkpoint = pl.callbacks.ModelCheckpoint(
                filepath=str(dir_checkpoints_experiment /
                             (f'seed={hparams.seed}' +
                              '_{epoch}_{val_loss:.3f}')),
                verbose=hparams.verbose,
                save_top_k=hparams.save_top_k,
                period=hparams.checkpoint_period,
            )

        # Early stopping callback
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=float(hparams.early_stop_min_delta),
            patience=hparams.early_stop_patience,
            verbose=hparams.verbose,
            mode=hparams.early_stop_mode,
        )

        # Define the trainer
        trainer = pl.Trainer(
            checkpoint_callback=checkpoint,
            max_epochs=hparams.epochs,
            logger=logger,
            val_check_interval=hparams.val_check_interval,
            gpus=hparams.gpus,
            early_stop_callback=early_stop_callback,
        )

        # Verbose messaging
        if hparams.verbose:
            now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f'\nCurrent time: {now}', flush=True)
            print(f'\nRunning with following hparams:', flush=True)
            pprint(hparams._content)

        # Define the model
        model = Model(hparams)
        if hparams.verbose:
            print(f'\nModel being used: \n{model}', flush=True)

        # Define the datamodule
        datamodule = datasets.DataModuleConstructor(hparams, Dataset)
 
        # Train the model
        print('\nBeginning training:', flush=True)
        now = datetime.datetime.now()
        trainer.fit(model, datamodule=datamodule)
        if hparams.verbose:
            elapsed = datetime.datetime.now() - now
            elapsed_fstr = time.strftime('%H:%M:%S', time.gmtime(
                elapsed.seconds))
            print(f'\nTraining completed! Time Elapsed: {elapsed_fstr}',
                  flush=True)

        
    else:
        raise NotImplementedError
        # # Get all the experiments with the name hparams.name*
        # experiments = list(index.DIR_CHECKPOINTS.glob(
        #     f'{hparams.name}_{hparams.exp_name}*'))

        # # import pdb; pdb.set_trace()
        # if len(experiments) > 1:
        #     # Get the newest exp by v number
        #     experiment_newest = sorted(
        #         experiments,
        #         key=lambda path: int(path.stem.split('_')[-1][1:]))[-1]
        #     # Get the model with the best (lowest) val_loss
        # else:
        #     experiment_newest = experiments[0]
        # experiment_newest_best_val = sorted(
        #     experiment_newest.iterdir(),
        #     key=lambda path: float(
        #         path.stem.split('val_loss=')[-1].split('_')[0]))[0]

        # model = Model.load_from_checkpoint(str(experiment_newest_best_val))
        # model.logger = logger
        # ## LOOK AT THIS LATER
        # model.prepare_data(val_path=const.DEFAULT_PATH)

        # # Define the trainer
        # trainer = pl.Trainer(
        #     logger=model.logger,
        #     gpus=hparams.gpus,
        #     max_epochs=1,
        # )

    if not hparams.no_test:
        # Ensure we are in cuda for testing if specified
        if 'cuda' in hparams.device and torch.cuda.is_available():
            model.cuda()

        # Create the test data
        test_data = np.array([datamodule.ds.array_data[n]
                              for n in const.DEFAULT_PATH]).reshape((
                                      1, len(const.DEFAULT_PATH), 2048))
        torch_data = torch.Tensor(test_data)

        # Get the model outputs
        outs = model.forward(torch_data, output_mode='eval')
        outs.update({'errors' : model.forward(torch_data, output_mode='error')})
        
        # Visualize the test data
        # import ipdb; ipdb.set_trace()
        figs = model.visualize(outs, borders=const.DEFAULT_BORDERS)
        if not hparams.no_graphs:
            for name, fig in figs.items():
                model.logger.experiment.log_image(name, fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipdb', action='store_true')
    parser.add_argument('--test_run', action='store_true')    

    # Check if we should have ipdb 
    temp_args, _ = parser.parse_known_args()

    # If running with test_run or ipdb
    if temp_args.ipdb or temp_args.test_run:
        with ipdb.launch_ipdb_on_exception():
            main(parser)
    else:
        main(parser)
