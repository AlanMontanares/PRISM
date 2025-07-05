import torch
import random

import numpy as np
import lightning as L

from torch.utils.data import DataLoader
from utils.custom_datasets import DelightClassic, DelightAutoLabeling


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DelightDataModule(L.LightningDataModule):

    def __init__(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        radius_train = None,
        ab_train = None,
        phi_train = None,
        batch_size=40,
        seed=0,
        num_workers=4,
        train_dataset_type="delight_classic",
    ):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.radius_train = radius_train
        self.ab_train = ab_train
        self.phi_train = phi_train

        self.batch_size = batch_size
        self.seed = seed

        self.num_workers = num_workers
        self.train_dataset_type = train_dataset_type

        self.persistent = num_workers > 0 and torch.cuda.is_available()

    def setup(self, stage=None):

        if self.train_dataset_type == "delight_classic":
            self.train_dataset = DelightClassic(self.X_train, self.y_train)

        elif self.train_dataset_type == "delight_autolabeling":
            self.train_dataset = DelightAutoLabeling(
                self.X_train,
                self.y_train,
                self.radius_train,
                self.ab_train,
                self.phi_train
            )
        
        else:
            raise ValueError(f"Unknown train_dataset_type: {self.train_dataset_type}")

        self.val_dataset = DelightClassic(self.X_val, self.y_val)
        self.test_dataset = DelightClassic(self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent,
            pin_memory=False,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
        )
