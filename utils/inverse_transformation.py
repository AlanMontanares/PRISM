import numpy as np
import torch

def revert_all_transforms(preds):
    """
    Revierte las transformaciones aplicadas a las imagenes pero en las posiciones predichas
    """
    # Tamaño de imagen
    H, W = 30, 30
    center = np.array([W/2 - 0.5, H/2 - 0.5])  # Centro de rotación real (14.5, 14.5)

    # Centros de cada transformación
    centers = np.array([
        [14,14],[14,15],[15,15],[15,14],
        [15,14],[14,14],[14,15],[15,15]
    ])  # (8, 2)

    # Matrices de transformación inversa (rotación + flip) - definidas con precisión
    def rot_matrix(deg):
        rad = np.deg2rad(deg)
        return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

    R0   = np.eye(2)
    R90  = rot_matrix(90)
    R180 = rot_matrix(180)
    R270 = rot_matrix(270)
    FH   = np.array([[-1, 0], [0, 1]])

    # Inversas (aplicar en sentido opuesto a las que usaste al transformar)
    transforms = np.stack([
        R0,
        R90,
        R180,
        R270,
        FH,
        FH @ R90,
        FH @ R180,
        FH @ R270,
    ])  # shape (8, 2, 2)

    # Convertir preds a numpy y sumar centros
    pred_np = preds.numpy() + centers  # (N, 8, 2)

    # Restar centro para rotar respecto al centro de la imagen
    rel_coords = pred_np - center  # (N, 8, 2)

    # Aplicar rotaciones/flips (vectorizado)
    reverted_rel = np.einsum("tij,ntj->nti", transforms, rel_coords)  # (N, 8, 2)

    # Volver al sistema original
    reverted_coords = reverted_rel + center  # (N, 8, 2)

    return torch.tensor(reverted_coords, dtype=preds.dtype)