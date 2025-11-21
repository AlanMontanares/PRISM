import torch
from torch.utils.data import Dataset

class DelightClassic(Dataset):
    def __init__(self, imgs, sn_pos):
        """
        imgs: Tensor de forma (batch, n_levels, n_channels, alto, ancho)
        sn_pos: Tensor de forma (batch, 2)
        """
        self.imgs = imgs
        self.sn_pos = sn_pos
        self.num_transforms = 8

    def __len__(self):
        return len(self.imgs)

    def _rotate_and_flip_image(self, img, k, flip):
        """
        img: Tensor de forma (n_levels, n_channels, alto, ancho)
        Retorna: Tensor transformado con la misma forma
        """
        # Los ejes espaciales son el alto (dim 2) y el ancho (dim 3)
        H_DIM = 2
        W_DIM = 3

        # 1. Flip horizontal (sobre el eje ancho)
        if flip:
            img = torch.flip(img, dims=[W_DIM])

        # 2. Rotación 90° k veces sobre los ejes alto y ancho
        img_rotated = torch.rot90(img, k=k, dims=[H_DIM, W_DIM])
        return img_rotated

    def _rotate_and_flip_keypoints(self, keypoints, k, flip, img_size=30):
        # Esta lógica no cambia, ya que solo depende de x, y
        x, y = keypoints
        if flip:
            x = img_size - 1 - x

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
        # Obtener la imagen original: (n_levels, n_channels, alto, ancho)
        image = self.imgs[idx]
        keypoints = self.sn_pos[idx] + 14

        images_transformed = []
        keypoints_transformed = []
        
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
            # Aplicar transformaciones usando los nuevos índices de dimensión
            transformed_img = self._rotate_and_flip_image(image, k, flip)
            images_transformed.append(transformed_img)

            transformed_keypoint = self._rotate_and_flip_keypoints(keypoints, k, flip)
            keypoints_transformed.append(transformed_keypoint)

        # Apilar las imágenes transformadas:
        # (8, n_levels, n_channels, alto, ancho)
        images_stack = torch.stack(images_transformed, dim=0).float()
        
        # ¡IMPORTANTE! El formato de salida es el deseado, no requiere permutación adicional.
        # Formato final: (augmentations, n_levels, n_channels, alto, ancho)

        # Apilar keypoints y aplicar corrección final (igual que antes)
        keys_stack = (
            torch.stack(keypoints_transformed).float() 
            - torch.tensor([
                [14,14],[14,15],[15,15],[15,14],
                [15,14],[14,14],[14,15],[15,15]
            ])
        )

        return images_stack, keys_stack


class RedshiftDataset(Dataset):
    def __init__(self, imgs, z):
        """
        imgs: Tensor de forma (batch, n_levels, n_channels, alto, ancho)
        z: Tensor de forma (batch, 1)
        """
        self.imgs = imgs
        self.z = z

    def __len__(self):
        return len(self.z)

    def _rotate_and_flip_image(self, img, k, flip):
        """
        img: Tensor de forma (n_levels, n_channels, alto, ancho)
        Retorna: Tensor transformado con la misma forma
        """
        # Los ejes espaciales son el alto (dim 2) y el ancho (dim 3)
        H_DIM = 2
        W_DIM = 3

        # 1. Flip horizontal (sobre el eje ancho)
        if flip:
            img = torch.flip(img, dims=[W_DIM])

        # 2. Rotación 90° k veces sobre los ejes alto y ancho
        img_rotated = torch.rot90(img, k=k, dims=[H_DIM, W_DIM])
        return img_rotated
    
    def __getitem__(self, idx):
        # Obtener la imagen original: (n_levels, n_channels, alto, ancho)
        image = self.imgs[idx]
        redshift = self.z[idx] 

        images_transformed = []
        redshift_repeated = []

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
            # Aplicar transformaciones usando los nuevos índices de dimensión
            transformed_img = self._rotate_and_flip_image(image, k, flip)
            images_transformed.append(transformed_img)

            redshift_repeated.append(redshift)

        # Apilar las imágenes transformadas:
        # (8, n_levels, n_channels, alto, ancho)
        images_stack = torch.stack(images_transformed, dim=0).float()
        redshift_stack = torch.stack(redshift_repeated).float()

        return images_stack, redshift_stack

class MultitaskDataset(Dataset):
    def __init__(self, imgs, sn_pos, z):
        """
        imgs: Tensor de forma (batch, n_levels, n_channels, alto, ancho)
        sn_pos: Tensor de forma (batch, 2)
        z: Tensor de forma (batch, 1)
        """
        self.imgs = imgs
        self.sn_pos = sn_pos
        self.z = z

    def __len__(self):
        return len(self.z)

    def _rotate_and_flip_image(self, img, k, flip):
        """
        img: Tensor de forma (n_levels, n_channels, alto, ancho)
        Retorna: Tensor transformado con la misma forma
        """
        # Los ejes espaciales son el alto (dim 2) y el ancho (dim 3)
        H_DIM = 2
        W_DIM = 3

        # 1. Flip horizontal (sobre el eje ancho)
        if flip:
            img = torch.flip(img, dims=[W_DIM])

        # 2. Rotación 90° k veces sobre los ejes alto y ancho
        img_rotated = torch.rot90(img, k=k, dims=[H_DIM, W_DIM])
        return img_rotated

    def _rotate_and_flip_keypoints(self, keypoints, k, flip, img_size=30):
        # Esta lógica no cambia, ya que solo depende de x, y
        x, y = keypoints
        if flip:
            x = img_size - 1 - x

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
        # Obtener la imagen original: (n_levels, n_channels, alto, ancho)
        image = self.imgs[idx]
        keypoints = self.sn_pos[idx] + 14
        redshift = self.z[idx] 

        images_transformed = []
        keypoints_transformed = []
        redshift_repeated = []

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
            # Aplicar transformaciones usando los nuevos índices de dimensión
            transformed_img = self._rotate_and_flip_image(image, k, flip)
            images_transformed.append(transformed_img)

            transformed_keypoint = self._rotate_and_flip_keypoints(keypoints, k, flip)
            keypoints_transformed.append(transformed_keypoint)

            redshift_repeated.append(redshift)

        # Apilar las imágenes transformadas:
        # (8, n_levels, n_channels, alto, ancho)
        images_stack = torch.stack(images_transformed, dim=0).float()
        redshift_stack = torch.stack(redshift_repeated).float()

        keys_stack = (
            torch.stack(keypoints_transformed).float() 
            - torch.tensor([
                [14,14],[14,15],[15,15],[15,14],
                [15,14],[14,14],[14,15],[15,15]
            ])
        )

        return images_stack, keys_stack, redshift_stack