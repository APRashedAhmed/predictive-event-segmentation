import logging
import argparse
from pathlib import Path
from pprint import pprint

import IPython
import pytorch_lightning as pl

from prevseg import index
from prevseg.dataloaders import breakfast
from prevseg.models import prednet

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--time_steps', type=int, default=64)
    parser.add_argument('--dir_checkpoints', type=str,
                        default=str(index.DIR_CHECKPOINTS))
    parser.add_argument('--dir_weights', type=str,
                        default=str(index.DIR_WEIGHTS))
    parser.add_argument('--dir_logs', type=str,
                        default=str(index.DIR_LOGS_TB))
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output_mode', type=str, default='error')
    parser.add_argument('--n_val', type=int, default=256)
    parser.add_argument('--n_test', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=117)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default='PredNet')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--dataloader', type=str,
                        default='BreakfastI3DFVDataset')
    parser.add_argument('--checkpoint_period', type=float, default=1.0)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--save_top_k', type=float, default=5)
    parser.add_argument('--gpus', type=float, default=1)
    parser.add_argument('--layer_loss_mode', type=str, default='first')
    parser.add_argument('--mini', type=bool, default=False)
    
    # add all the available options to the trainer
    # parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    # Make sure this is correct
    if hasattr(prednet, hparams.model):
        Model = getattr(prednet, hparams.model)
        hparams.name = Model.name if not hparams.name else hparams.name
    else:
        raise Exception(f'Invalid model "{hparams.model}" passed.')

    # Check this is correct as well
    if hasattr(breakfast, hparams.dataloader):
        Dataloader = getattr(breakfast, hparams.dataloader)
    else:
        raise Exception(f'Invalid dataloader "{hparams.dataloader}" passed.')

    # Tensorboard logger
    log_dir = Path(hparams.dir_logs) / f'{hparams.name}'
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    logger = pl.loggers.TensorBoardLogger(str(log_dir.parent),
                                          name=hparams.name)

    # Checkpoint Call back
    ckpt_dir = Path(hparams.dir_checkpoints) \
        / f'{hparams.name}_v{logger.version}'
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True)
    ckpt = pl.callbacks.ModelCheckpoint(
        filepath=str(
            ckpt_dir / 'bk_i3d_{global_step:05d}_{epoch:03d}_{val_loss:.3f}'),
        verbose=True,
        save_top_k=hparams.save_top_k,
        period=hparams.checkpoint_period,
    )

    # Define the trainer
    trainer = pl.Trainer(
        default_save_path=str(index.DIR_CHECKPOINTS),
        checkpoint_callback=ckpt,
        max_epochs=hparams.epochs,
        logger=logger,
        val_check_interval=hparams.val_check_interval,
        gpus=hparams.gpus,
    )

    print(f'Running with following hparams:')
    pprint(vars(hparams))

    # Define the model
    model = Model(hparams)
    print(model, flush=True)

    print('Loading data')
    ds = Dataloader()
    model.ds = ds
    
    print('Beginning training')
    # Train the model
    trainer.fit(model)

    IPython.embed()
