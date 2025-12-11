import torch
import random

import numpy as np
import lightning as L

from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.custom_datasets import DelightClassic, RedshiftDataset, MultitaskDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class PRISMDataModule(L.LightningDataModule):

    def __init__(
        self,
        X_train,
        X_val,
        X_test,
        X_val_pos,
        X_val_z,
        X_test_pos,
        X_test_z,
        pos_train,
        pos_val,
        pos_test,
        z_train,
        z_val,
        z_test,
        batch_size=40,
        seed=0,
        num_workers=4,
        task = "galaxy_hunter",
        use_sampler=False,
    ):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.X_val_pos = X_val_pos
        self.X_val_z = X_val_z

        self.X_test_pos = X_test_pos
        self.X_test_z = X_test_z

        self.pos_train = pos_train
        self.pos_val = pos_val
        self.pos_test = pos_test

        self.z_train = z_train
        self.z_val = z_val
        self.z_test = z_test

        self.batch_size = batch_size
        self.seed = seed

        self.num_workers = num_workers
        self.persistent = num_workers > 0 and torch.cuda.is_available()

        self.task = task
        self.use_sampler=use_sampler

    def setup(self, stage=None):

        if self.task == "galaxy_hunter":
            self.train_dataset = DelightClassic(self.X_train, self.pos_train)
            self.val_dataset = DelightClassic(self.X_val, self.pos_val)
            self.test_dataset = DelightClassic(self.X_test, self.pos_test)

        elif self.task == "redshift_prediction":
            self.train_dataset = RedshiftDataset(self.X_train, self.z_train)
            self.val_dataset = RedshiftDataset(self.X_val, self.z_val)
            self.test_dataset = RedshiftDataset(self.X_test, self.z_test)
        
        elif self.task == "multitask":
            self.train_dataset = MultitaskDataset(self.X_train, self.pos_train, self.z_train)

            self.val_dataset_pos = DelightClassic(self.X_val_pos, self.pos_val)
            self.val_dataset_redshift = RedshiftDataset(self.X_val_z, self.z_val)

            self.test_dataset_pos = DelightClassic(self.X_test_pos, self.pos_test) #a
            self.test_dataset_redshift = RedshiftDataset(self.X_test_z, self.z_test)

    def train_dataloader(self):

        if self.use_sampler:

            distances = np.linalg.norm(self.pos_train, axis=1)*0.25 #arcsec

            mask_large = distances > 50
            mask_small = ~mask_large

            n_large = mask_large.sum()
            n_small = mask_small.sum()

            w_large = 1.0 / n_large 
            w_small = 1.0 / n_small 

            sample_weights = np.zeros_like(distances, dtype=np.float64)
            sample_weights[mask_large] = w_large
            sample_weights[mask_small] = w_small

            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True
            )

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler if self.use_sampler else None,
            shuffle= not self.use_sampler,
            num_workers=self.num_workers,
            persistent_workers=self.persistent,
            pin_memory=False,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self):

        if self.task == "multitask":
            val_loader_task1 = DataLoader(
                self.val_dataset_pos,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent,
                pin_memory=False,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(self.seed),
            )
            val_loader_task2 = DataLoader(
                self.val_dataset_redshift,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent,
                pin_memory=False,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(self.seed),
            )
            return [val_loader_task1, val_loader_task2]

        else:
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

    def test_dataloader(self):
        if self.task == "multitask":
            test_loader_task1 = DataLoader(
                self.test_dataset_pos,
                batch_size=40,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent,
                pin_memory=False,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(self.seed),
            )
            test_loader_task2 = DataLoader(
                self.test_dataset_redshift,
                batch_size=40,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent,
                pin_memory=False,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(self.seed),
            )
            return [test_loader_task1, test_loader_task2]

        else:
            return DataLoader(
                self.test_dataset,
                batch_size=40,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent,
                pin_memory=False,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(self.seed),
            )
