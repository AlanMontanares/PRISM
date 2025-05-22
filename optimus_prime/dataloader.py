import torch
import random

import numpy as np
import lightning as L

from torch.utils.data import DataLoader, Dataset
import albumentations as A

from utils.sersic_functions import sersic_profile, generate_random_pos

class RecenterOnSersicSN(A.ImageOnlyTransform):
    def __init__(self, sn_pos):
        super().__init__(p=1.0)
        self.sn_pos = np.array(sn_pos) 

    def apply(self, image, **params):

        centered = self.center_on_sn(image, self.sn_pos)
        return centered

    def center_on_sn(self, multires_images, sn_pos):
        center_images = []
        for i in range(5):
            img = multires_images[:,:,i]
            x_center = int(round(-(sn_pos[0] / (2**i)) + 135))
            y_center = int(round(-(sn_pos[1] / (2**i)) + 135))

            x_min = max(0, x_center - 15)
            y_min = max(0, y_center - 15)
            x_max = min(img.shape[1], x_center + 15) # el eje x esta en la dim 1
            y_max = min(img.shape[0], y_center + 15) # el eje y esta en la dim 0

            crop = A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
            cropped = crop(image=img)["image"]
            center_images.append(cropped)

        return np.transpose(np.array(center_images), (1, 2, 0))
    

class DelighDataset(Dataset):
    def __init__(self, imgs, sn_pos, sersic_radius, sersic_ab, sersic_phi, augmentation="delight"):

        self.imgs = imgs
        self.sn_pos = sn_pos
        self.sersic_radius = sersic_radius
        self.sersic_ab = sersic_ab
        self.sersic_phi = sersic_phi
        self.augmentation = augmentation

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        image = self.imgs[idx]

        if self.augmentation == "classic":
            keypoints = self.sn_pos[idx][::-1] + 15 # Originalmente es x,y con [::-1] lo invertimos

            transform = A.Compose(
                [
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.pytorch.ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )

            transformed = transform(image=image, keypoints=[keypoints])
            image = transformed["image"]
            keypoints = transformed["keypoints"]

            return image.unsqueeze(1).float(), torch.tensor(keypoints[0]).float() - 15 #Lo centramos nuevamente
        
        elif self.augmentation == "auto-labeling":

            ser_radio = self.sersic_radius[idx]
            ser_ab = self.sersic_ab[idx]
            ser_phi = self.sersic_phi[idx] 

            auto_sn_pos = generate_random_pos(sersic_radius=ser_radio,
                                                sersic_ab=ser_ab,
                                                sersic_phi=ser_phi)[::-1] # lo dejamos en x,y como lo necesita albumentations

            auto_label_transform = RecenterOnSersicSN(sn_pos=auto_sn_pos) # Para centrar en el auto_sn_pos

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
                composed = A.Compose(
                    [auto_label_transform, t, A.pytorch.ToTensorV2()], # Igual a Delight pero ahora con auto-etiquetado
                    keypoint_params=A.KeypointParams(
                        format="xy", remove_invisible=False
                    ),
                )
                transformed = composed(image=image, keypoints=[auto_sn_pos + 15]) # le sumamos el centro porque asi lo requiere albumentations
                images.append(transformed["image"])
                keys.append(transformed["keypoints"][0])

            return torch.stack(images).unsqueeze(2).float(), torch.tensor(keys).float() - 15 # le restamos el centro porque asi lo requiere DELIGHT


        elif self.augmentation == "delight":

            keypoints = self.sn_pos[idx][::-1] + 15 # Originalmente es x,y con [::-1] lo invertimos

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
                composed = A.Compose(
                    [t, A.pytorch.ToTensorV2()],
                    keypoint_params=A.KeypointParams(
                        format="xy", remove_invisible=False
                    ),
                )
                transformed = composed(image=image, keypoints=[keypoints])
                images.append(transformed["image"])
                keys.append(transformed["keypoints"][0])

            return torch.stack(images).unsqueeze(2).float(), torch.tensor(keys).float() -15

        elif self.augmentation == "None":
            
            keypoints = self.sn_pos[idx][::-1] + 15 # Originalmente es x,y con [::-1] lo invertimos

            transform = A.Compose(
                [A.pytorch.ToTensorV2()],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )

            transformed = transform(image=image, keypoints=[keypoints])
            image = transformed["image"]
            keypoints = transformed["keypoints"]

            return image.unsqueeze(1).float(), torch.tensor(keypoints[0]).float() -15


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
        radius_train,
        radius_val,
        radius_test,
        ab_train,
        ab_val,
        ab_test,
        phi_train,
        phi_val,
        phi_test,
        batch_size=128,
        seed=0,
        num_workers=4,
        train_augmentation="delight",
    ):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.radius_train = radius_train
        self.radius_val = radius_val
        self.radius_test = radius_test

        self.ab_train = ab_train
        self.ab_val = ab_val
        self.ab_test = ab_test

        self.phi_train = phi_train
        self.phi_val = phi_val
        self.phi_test = phi_test

        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.train_augmentation = train_augmentation

        self.persistent = num_workers > 0 and torch.cuda.is_available()

    def setup(self, stage=None):

        self.train_dataset = DelighDataset(
            self.X_train,
            self.y_train,
            self.radius_train,
            self.ab_train,
            self.phi_train,
            augmentation=self.train_augmentation,
        )

        self.val_dataset = DelighDataset(
            self.X_val,
            self.y_val,
            self.radius_val,
            self.ab_val,
            self.phi_val,
            augmentation = (
                "delight"
                if self.train_augmentation == "delight" or self.train_augmentation == "auto-labeling"
                else "None" # en caso de classic augmentation
            )
        )

        self.test_dataset = DelighDataset(
            self.X_test,
            self.y_test,
            self.radius_test,
            self.ab_test,
            self.phi_test,
            augmentation = (
                "delight"
                if self.train_augmentation == "delight" or self.train_augmentation == "auto-labeling"
                else "None" # en caso de classic augmentation
            )
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent,
            pin_memory=False,
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
