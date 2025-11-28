import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
import time
import shutil

from astroquery.hips2fits import hips2fits

import argparse
import os
from joblib import Parallel, delayed

from .h2f_download_functions import get_augmented_multires
from utils.sersic_functions import sersic_profile


def augment_dataframe(data_frame, percentaje=0.01, n_jobs=-1):
    """
    Duplica filas de un DataFrame según el número de augmentaciones calculado
    con un perfil de Sérsic.
    """
    def compute_num_aug(row):
        sersic_img = sersic_profile(
            image_shape=(600, 600),
            x_center=299, y_center=299,
            Re_arcsec=row["rSerRadius"],
            b_over_a=row["rSerAb"],
            theta_deg=row["rSerPhi"],
            pixel_scale=0.25,
            Ie=1.0,
            n=4
        )
        n_pix = np.count_nonzero(sersic_img) * percentaje
        return int(np.ceil(n_pix))

    nums = Parallel(n_jobs=n_jobs)(
        delayed(compute_num_aug)(row) for _, row in data_frame.iterrows()
    )
    data_frame = data_frame.copy()
    data_frame["num_augmentations"] = nums

    data_frame_aug = data_frame.loc[
       data_frame.index.repeat(data_frame["num_augmentations"])
    ].reset_index(drop=True)

    return data_frame_aug



def download_batch(data_frame, img_size, inicio, final, name_dataset, filters):
    """
    Descarga un batch de imágenes multiresolución y posiciones de supernovas
    simuladas, y guarda los resultados en un archivo `.npz`.

    Para cada fila del DataFrame en el rango [inicio, final), se generan 
    `num_augmentations` posiciones de supernovas usando `get_augmented_multires`.
    En caso de error, se rellenan los arrays con ceros.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        DataFrame que contiene la información de las galaxias.
    img_size : int
        Tamaño en píxeles de las imágenes a generar.
    inicio : int
        Índice inicial (incluido) del batch.
    final : int
        Índice final (excluido) del batch.
    name_dataset : str
        Nombre de la carpeta y prefijo del dataset donde se guardará el archivo.

    Notes
    -----
    - Cada imagen multiresolución se guarda como un array de forma 
      (num_augmentations, 5, img_size, img_size), donde 5 corresponde
      a los niveles de resolución.
    - Las posiciones se guardan como arrays de forma (num_augmentations, 2).
    - El archivo se guarda como `{name_dataset}/{name_dataset}_{final}.npz`.
    """
    img_stack = []
    pos_stack = []
    max_retry = 4

    for x in tqdm(range(inicio,final)):
        
        num_augmentations = 1

        for retry in range(max_retry):
            try:
                
                img, pos = get_augmented_multires(data_frame, x, size=img_size, num_augmentations=num_augmentations, filters=filters)
                img_stack.append(img)
                pos_stack.append(pos)
                break

            except:
                if retry+1 == max_retry:
                    img_stack.append(np.zeros((5,len(filters),img_size,img_size), dtype=np.float32))
                    pos_stack.append(np.zeros((1,2), dtype=np.float32))


    np.savez(f'{name_dataset}/{name_dataset}_{final}.npz', imgs=np.concatenate(img_stack), pos=np.concatenate(pos_stack))

def download_all(df, img_size, name_dataset, n_procesos, filters):
    """
    Descarga todas las imágenes multiresolución y posiciones de supernovas
    de un DataFrame en batches y concatena los resultados en un único 
    archivo `.npz` final.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con la información de las galaxias.
    img_size : int
        Tamaño en píxeles de las imágenes a generar.
    name_dataset : str
        Nombre del dataset. También se usa como nombre de la carpeta de salida.
    n_procesos : int
        Número de hilos a utilizar

    Returns
    -------
    None
        Se guarda un archivo `.npz` final con todas las imágenes y posiciones concatenadas.

    Notes
    -----
    - Cada batch se guarda temporalmente como `{name_dataset}/{name_dataset}_{batch}.npz`.
    - Al final se concatenan todos los batches en un único array:
        - `full_imgs` de forma (N, img_size, img_size, 5)
        - `full_pos` de forma (N, 2)
      donde `N` es el número total de galaxias en `df`.
    - El archivo final se guarda en `../data/SERSIC/X_train_{name_dataset}.npz`.
    - Se usa `threading` para paralelizar la descarga de batches.
    """
    os.makedirs(name_dataset, exist_ok=True)
    os.makedirs('data/SERSIC', exist_ok=True)

    start = 0
    total = len(df)

    arr = np.linspace(start, total, n_procesos+1, dtype=int)

    threads = []

    for i in range(n_procesos):
        t = threading.Thread(target=download_batch, args=[df, img_size, arr[i], arr[i+1], name_dataset, filters])
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()


    full_imgs = []
    full_pos = []
    for batch in arr[1:]:
        file = np.load(f'{name_dataset}/{name_dataset}_{batch}.npz')
        full_imgs.append(file["imgs"])
        full_pos.append(file["pos"])

    full_imgs = np.concatenate(full_imgs, axis=0)
    full_pos = np.concatenate(full_pos, axis=0)

    np.savez(f'data/SERSIC/X_train_{name_dataset}.npz', imgs=full_imgs, pos=full_pos)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe_path', type=str, default="data/SERSIC/df_train_delight.csv", help='Ruta al dataframe')
    parser.add_argument('--img_size', type=int, default=30, help='Tamaño de las imagenes a descargar')
    parser.add_argument('--filters', type=str, default='r', help='Filtros a descargar')
    parser.add_argument('--percentaje', type=float, default=None, help='Porciento de augmentations')
    parser.add_argument('--n_procesos', type=int, default=8, help='Numero de hilos')
    parser.add_argument('--name_dataset', type=str, default="augmented_dataset", help='Nombre del dataset')
    parser.add_argument('--use_mirror', type=str, default=None, help="Usar el servidor mirror de Hips2FITS")

    args = parser.parse_args()

    if args.use_mirror:
        print("Usando el servidor mirror")
        hips2fits.server = 'https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits'

    df = pd.read_csv(args.dataframe_path)

    if args.percentaje:
        df = augment_dataframe(df, args.percentaje)

    alpha = time.time()
    download_all(df, args.img_size, args.name_dataset, args.n_procesos, args.filters)

    shutil.rmtree(args.name_dataset)

    print(f"Fin Total: {time.time()-alpha} [s]")