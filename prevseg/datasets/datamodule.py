"""Datamodules"""
import logging

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class DataModuleConstructor(pl.LightningDataModule):
    def __init__(self, hparams, Dataset, DatasetVal=None, ds=None, ds_val=None):
        super().__init__()
        self.hparams = hparams
        self.Dataset = Dataset
        self.ds = ds
        self.ds_val = ds
        self.prepare_data()
        self.setup_data()

    def prepare_data(self):
        self.Dataset.prepare_data(self.hparams, self)

    def setup_data(self, stage=None):
        self.Dataset.setup_data(self.hparams, self)

    def train_dataloader(self):
        return self.Dataset.train_dataloader(self.hparams, self, self.ds)

    def val_dataloader(self):
        return self.Dataset.val_dataloader(self.hparams, self, self.ds_val)
       
