import torch
import numpy as np

from torch.utils.data import Dataset
import albumentations as A

from utils.sersic_functions import generate_random_pos

def recenter_on_sn(image, sn_pos):
    center_images = []
    for i in range(5):
        img = image[:, :, i]
        x_center = 135 - round(-sn_pos[0] / (2**i)) # Para centrarla en ancho/2 - 1pix  usamos el ancho/2 
        y_center = 135 - round(-sn_pos[1] / (2**i)) 
        x_min = max(0, x_center - 15)
        y_min = max(0, y_center - 15)
        x_max = min(img.shape[1], x_center + 15)
        y_max = min(img.shape[0], y_center + 15)

        cropped = img[y_min:y_max, x_min:x_max]
        center_images.append(cropped)

    return np.transpose(np.array(center_images), (1, 2, 0))
    

# class DelightBasic(Dataset):
#     def __init__(self, imgs, sn_pos):

#         self.imgs = imgs
#         self.sn_pos = sn_pos


#         self.transform =  A.Compose(
#             [
#                 A.RandomRotate90(p=0.5),
#                 A.HorizontalFlip(p=0.5),
#                 A.pytorch.ToTensorV2(),
#             ],
#             keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
#         )

#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, idx):

#         image = self.imgs[idx]

#         keypoints = self.sn_pos[idx][::-1] + 14 # Originalmente es x,y con [::-1] lo invertimos

#         transformed = self.transform(image=image, keypoints=[keypoints])
#         image = transformed["image"]
#         keypoints = transformed["keypoints"]

#         return image.unsqueeze(1).float(), torch.tensor(keypoints[0]).float() - 14 #Lo centramos nuevamente


class DelightClassic(Dataset):
    def __init__(self, imgs, sn_pos):

        self.imgs = imgs
        self.sn_pos = sn_pos

        self.transforms = [
                            A.NoOp(),
                            A.Rotate(limit=(90, 90), p=1.0),
                            A.Rotate(limit=(180, 180), p=1.0),
                            A.Rotate(limit=(270, 270), p=1.0),
                            A.HorizontalFlip(p=1.0),
                            A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(90, 90), p=1.0)]),
                            A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(180, 180), p=1.0)]),
                            A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(270, 270), p=1.0)]),
                        ]
    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):

        image = self.imgs[idx]

        keypoints = self.sn_pos[idx][::-1] + 14 # Originalmente es x,y con [::-1] lo invertimos

        images = []
        keys = []

        for t in self.transforms:
            composed = A.Compose(
                [t, A.pytorch.ToTensorV2()],
                keypoint_params=A.KeypointParams(
                    format="xy", remove_invisible=False
                ),
            )
            transformed = composed(image=image, keypoints=[keypoints])
            images.append(transformed["image"])
            keys.append(transformed["keypoints"][0])

        return torch.stack(images).unsqueeze(2).float(), torch.tensor(keys).float() - torch.tensor([[14,14],[14,15],[15,15],[15,14],[15,14],[14,14],[14,15],[15,15]])



class DelightAutoLabeling(Dataset):
    def __init__(self, imgs, sn_pos, sersic_radius=None, sersic_ab=None, sersic_phi=None):

        self.imgs = imgs
        self.sn_pos = sn_pos
        self.sersic_radius = sersic_radius
        self.sersic_ab = sersic_ab
        self.sersic_phi = sersic_phi

        self.transforms = [
                            A.NoOp(),
                            A.Rotate(limit=(90, 90), p=1.0),
                            A.Rotate(limit=(180, 180), p=1.0),
                            A.Rotate(limit=(270, 270), p=1.0),
                            A.HorizontalFlip(p=1.0),
                            A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(90, 90), p=1.0)]),
                            A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(180, 180), p=1.0)]),
                            A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(270, 270), p=1.0)]),
                        ]
    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):

        image = self.imgs[idx]

        if self.sn_pos is not None:
            auto_sn_pos = self.sn_pos[idx][::-1] # Para centrar en el auto_sn_pos (en caso de tenerlos)

        else:

            ser_radio = self.sersic_radius[idx]
            ser_ab = self.sersic_ab[idx]
            ser_phi = self.sersic_phi[idx] 

            auto_sn_pos = generate_random_pos(sersic_radius=ser_radio,
                                                sersic_ab=ser_ab,
                                                sersic_phi=ser_phi)[::-1] # lo dejamos en x,y como lo necesita albumentations

        image = recenter_on_sn(image, auto_sn_pos)

        images = []
        keys = []

        for t in self.transforms:
            composed = A.Compose(
                [t, A.pytorch.ToTensorV2()], # Igual a Delight pero ahora con auto-etiquetado
                keypoint_params=A.KeypointParams(
                    format="xy", remove_invisible=False
                ),
            )
            transformed = composed(image=image, keypoints=[auto_sn_pos + 14]) # le sumamos el centro porque asi lo requiere albumentations
            images.append(transformed["image"])
            keys.append(transformed["keypoints"][0])

        return torch.stack(images).unsqueeze(2).float(), torch.tensor(keys).float() - torch.tensor([[14,14],[14,15],[15,15],[15,14],[15,14],[14,14],[14,15],[15,15]])