import torch
import random

import numpy as np
import lightning as L

from torch.utils.data import DataLoader, Dataset
import albumentations as A


class DelighDataset(Dataset):
    def __init__(self,imgs, sn_pos, augmentation="delight"):

        self.imgs = imgs
        self.sn_pos =sn_pos
        self.augmentation = augmentation

    def __len__(self):
        return len(self.sn_pos)

    def __getitem__(self, idx):

        image = self.imgs[idx]  
        keypoints = self.sn_pos[idx][::-1] + 15

        if self.augmentation == "data_augmentation":

            transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.pytorch.ToTensorV2()
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                
            transformed = transform(image=image, keypoints=[keypoints])
            image = transformed["image"]
            keypoints = transformed["keypoints"]

            return image, np.array(keypoints[0])
        
        elif self.augmentation == "delight":
            transforms = [
                A.NoOp(),
                A.Rotate(limit=(90, 90), p=1.0),
                A.Rotate(limit=(180, 180), p=1.0),
                A.Rotate(limit=(270, 270), p=1.0),
                A.HorizontalFlip(p=1.0),
                A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(90, 90), p=1.0)]),
                A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(180, 180), p=1.0)]),
                A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(270, 270), p=1.0)]),
            ]

            images = []
            keys = []

            for t in transforms:
                composed = A.Compose([t, A.pytorch.ToTensorV2()], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                transformed = composed(image=image, keypoints=[keypoints])
                images.append(transformed["image"])
                keys.append(transformed["keypoints"][0])

            return np.expand_dims(np.stack(images), axis=2), np.stack(keypoints)
        

        elif self.augmentation == "None":

            transform = A.Compose([
                A.pytorch.ToTensorV2()
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                
            transformed = transform(image=image, keypoints=[keypoints])
            image = transformed["image"]
            keypoints = transformed["keypoints"]

            return image, np.array(keypoints[0])
            

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class DelightDataModule(L.LightningDataModule):
    
    def __init__(self, X_train, X_val, X_test,
                 y_train, y_val, y_test, batch_size=128, seed=0, num_workers =4, train_augmentation="delight"):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test


        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.train_augmentation = train_augmentation

    def setup(self, stage=None):
        
        self.train_dataset = DelighDataset(
          self.X_train,
          self.y_train,
          augmentation=self.train_augmentation,
        )

        self.val_dataset = DelighDataset(
          self.X_val,
          self.y_val,
          augmentation = self.train_augmentation if self.train_augmentation == "delight" else "None"
        )


        self.test_dataset = DelighDataset(
          self.X_test,
          self.y_test,
          augmentation= self.train_augmentation if self.train_augmentation == "delight" else "None",
          )
        

    def train_dataloader(self):
        return DataLoader(
          self.train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=self.num_workers,
          persistent_workers=True,
          pin_memory=False,
          worker_init_fn=seed_worker,
          generator = torch.Generator().manual_seed(self.seed)
        )

    def val_dataloader(self):
        return DataLoader(
          self.val_dataset,
          batch_size=self.batch_size,
          shuffle=False,
          num_workers=self.num_workers,
          persistent_workers=True,
          pin_memory=False,
          generator = torch.Generator().manual_seed(self.seed)
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=False,
            generator = torch.Generator().manual_seed(self.seed)
        )