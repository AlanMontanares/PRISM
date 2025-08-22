import torch
import albumentations as A

from torch.utils.data import Dataset

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

        image = self.imgs[idx].numpy()

        keypoints = self.sn_pos[idx].numpy() + 14 

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


class DelightClassicOptimized(Dataset):
    def __init__(self, imgs, sn_pos):
        self.imgs = imgs
        self.sn_pos = sn_pos
        self.num_transforms = 8  # Número de transformaciones

    def __len__(self):
        return len(self.imgs)

    def _rotate_and_flip_image(self, img, k, flip):
        # Primero flip si corresponde
        if flip:
            img = torch.flip(img, dims=[1])  # horizontal flip
        # Luego rotar
        img_rotated = torch.rot90(img.permute(2,0,1), k=k, dims=[1,2]).permute(1,2,0)
        return img_rotated

    def _rotate_and_flip_keypoints(self, keypoints, k, flip, img_size=30):
        x, y = keypoints

        # Flip horizontal primero
        if flip:
            x = img_size - 1 - x

        # Rotación después
        if k == 0:  # 0 grados
            x_new, y_new = x, y
        elif k == 1:  # 90 grados 
            x_new = y
            y_new = img_size - 1 - x
        elif k == 2:  # 180 grados
            x_new = img_size - 1 - x
            y_new = img_size - 1 - y
        elif k == 3:  # 270 grados
            x_new = img_size - 1 - y
            y_new = x

        return torch.tensor([x_new, y_new])

    def __getitem__(self, idx):
        # Obtener la imagen y los keypoints originales
        image = self.imgs[idx]
        keypoints = self.sn_pos[idx] + 14

        # Lista para almacenar las imágenes y keypoints transformados
        images_transformed = []
        keypoints_transformed = []
        
        # Definir las transformaciones: (rotación_k, volteo_horizontal)
        transformations = [
            (0, False),  # NoOp
            (1, False),  # Rotate 90
            (2, False),  # Rotate 180
            (3, False),  # Rotate 270
            (0, True),   # HorizontalFlip
            (1, True),   # HorizontalFlip + Rotate 90
            (2, True),   # HorizontalFlip + Rotate 180
            (3, True),   # HorizontalFlip + Rotate 270
        ]

        for k, flip in transformations:
            # Aplicar transformaciones a la imagen
            transformed_img = self._rotate_and_flip_image(image, k, flip)
            images_transformed.append(transformed_img)

            # Aplicar transformaciones a los keypoints
            transformed_keypoint = self._rotate_and_flip_keypoints(keypoints, k, flip)
            keypoints_transformed.append(transformed_keypoint)

        # Apilar los resultados y ajustar las dimensiones
        # Las imágenes tienen shape (8, 30, 30, 5) -> se necesita (8, 5, 30, 30) para PyTorch
        images_stack = torch.stack(images_transformed).permute(0, 3, 1, 2)
        images_stack = images_stack.float()
        
        # Apilar los keypoints
        keys_stack = torch.stack(keypoints_transformed).float() - torch.tensor([[14,14],[14,15],[15,15],[15,14],[15,14],[14,14],[14,15],[15,15]])

        return images_stack.unsqueeze(2), keys_stack
    