import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
import time
import shutil

import argparse
import os

from .h2f_download_functions import get_augmented_multires


def map_exponential(x, x_min=10, x_max=70, y_min=5, y_max=500):
    """
    Mapea un arreglo de valores a un rango exponencial con límites inferiores y superiores.

    Los valores menores a `x_min` se fijan en `x_min` y los mayores a `x_max` se fijan en `x_max`.
    Luego se normalizan al rango [0, 1] y se escalan exponencialmente entre `y_min` y `y_max`.
    El resultado final se devuelve como enteros.

    Parameters
    ----------
    x : array_like
        Arreglo de entrada con valores a transformar.
    x_min : float, optional
        Valor mínimo del rango de entrada. Todo valor menor se recorta a este límite.
    x_max : float, optional
        Valor máximo del rango de entrada. Todo valor mayor se recorta a este límite.
    y_min : float, optional
        Valor mínimo del rango de salida.
    y_max : float, optional
        Valor máximo del rango de salida.

    Returns
    -------
    numpy.ndarray
        Arreglo con los valores transformados en enteros.
    """
    # Clipping
    x = np.clip(x, x_min, x_max)
    # Normalizar a [0, 1]
    norm = (x - x_min) / (x_max - x_min)
    # Escalado exponencial
    val = y_min * (y_max / y_min) ** norm
    return val.astype(int)


def download_batch(data_frame, img_size, inicio, final, name_dataset):
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

    max_retry = 2
    for x in tqdm(range(inicio,final)):
        
        radio_sersic = data_frame.iloc[x]["rSerRadius"]
        num_augmentations = map_exponential(radio_sersic)

        for retry in range(max_retry):
            try:
                img, pos = get_augmented_multires(data_frame, x, size=img_size, num_augmentations=num_augmentations)
                img_stack.append(img)
                pos_stack.append(pos)
                break

            except:
                if retry+1 == max_retry:
                    img_stack.append(np.zeros((num_augmentations, 5, img_size, img_size), dtype=np.float32))
                    pos_stack.append(np.zeros((num_augmentations, 2), dtype=np.float32))

    np.savez(f'{name_dataset}/{name_dataset}_{final}.npz', imgs=np.concatenate(img_stack), pos=np.concatenate(pos_stack))

def download_all(df, img_size, name_dataset, batch_size):
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
    batch_size : int
        Número de ejemplos por batch. Cada batch se descarga en un hilo independiente.

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
    os.makedirs('data\SERSIC', exist_ok=True)

    start = 0
    total = len(df)

    arr = np.arange(start, total, batch_size)
    arr = np.append(arr, total)

    n_procesos = len(arr)-1
    threads = []


    for i in range(n_procesos):
        stop = min(arr[i]+batch_size, total)
        t = threading.Thread(target=download_batch, args=[df, img_size, arr[i], stop, name_dataset])
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()


    full_imgs = []
    full_pos = []
    for batch in arr[1:n_procesos+1]:
        file = np.load(f'{name_dataset}/{name_dataset}_{batch}.npz')
        full_imgs.append(file["imgs"])
        full_pos.append(file["pos"])

    full_imgs = np.concatenate(full_imgs, axis=0)
    full_imgs = np.transpose(full_imgs, (0, 2, 3, 1))


    full_pos = np.concatenate(full_pos, axis=0)

    np.savez(f'data\SERSIC\X_train_{name_dataset}.npz', imgs=full_imgs, pos=full_pos)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe_path', type=str, default="..\data\SERSIC\df.csv", help='Ruta al dataframe')
    parser.add_argument('--img_size', type=int, default=30, help='Tamaño de las imagenes a descargar')
    parser.add_argument('--batch_size', type=int, default=100, help='Tamaño del batch para la descargar')
    parser.add_argument('--name_dataset', type=str, default="augmented_dataset", help='Nombre del dataset')

    args = parser.parse_args()

    df = pd.read_csv(args.dataframe_path)

    alpha = time.time()

    download_all(df, args.img_size, args.name_dataset, args.batch_size)

    shutil.rmtree(args.name_dataset)

    print(f"Fin Total: {time.time()-alpha} [s]")