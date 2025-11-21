import argparse
import os
import shutil
import threading
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


from .h2f_download_functions import get_multires



def download_batch(data_frame, img_size, inicio, final, name_dataset, filters):
    """
    Descarga un batch de imágenes multiresolución de galaxias y lo guarda en un directorio.

    Para cada fila del DataFrame en el rango [inicio, final), se obtiene una imagen
    multiresolución usando `get_multires`. En caso de error, se rellena con ceros.
    Los resultados se guardan en un archivo `.npy`.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        DataFrame que contiene la información de las galaxias.
    img_size: int
        Tamaño de las imagenes a descargar
    inicio : int
        Índice inicial (incluido) del batch.
    final : int
        Índice final (excluido) del batch.
    name_dataset : str
        Nombre de la carpeta y prefijo del dataset donde se guardará el archivo.

    Notes
    -----
    - Cada ejemplo se guarda como un array de forma (5, size, size), con `size=30`.
    - El archivo se guarda como: `{name_dataset}/{name_dataset}_{final}.npy`.
    - Se permiten hasta 2 reintentos (`max_retry=2`) por imagen.
    """

    stack = []
    max_retry = 4

    for x in tqdm(range(inicio,final)):

        for retry in range(max_retry):
            try:
                img = get_multires(data_frame, x, size=img_size, filters=filters)
                stack.append(img)
                break

            except:
                if retry+1 == max_retry:
                    stack.append(np.zeros((5,len(filters),img_size,img_size), dtype=np.float32))
    
    np.save(f'{name_dataset}/{name_dataset}_{final}.npy', np.stack(stack))


def download_all(df, img_size, name_dataset, n_procesos, filters):
    """
    Descarga todas las imágenes multiresolución de un DataFrame en batches 
    y las combina en un único archivo final.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con la información de las galaxias.
    img_size: int
        Tamaño de las imagenes a descargar
    name_dataset : str
        Nombre del dataset. También se usa como nombre de la carpeta de salida.
    n_procesos : int
        Número de hilos a utilizar

    Returns
    -------
    None
        Se guarda un archivo `.npy` con todas las imágenes concatenadas.

    Notes
    -----
    - Cada batch se guarda temporalmente como `{name_dataset}/{name_dataset}_{batch_end}.npy`.
    - Al final se concatenan todos los batches en un único array de forma (N, img_size, img_size, 5),
      donde `N` es el número total de galaxias en `df`.
    - El archivo final se guarda en: `../data/SERSIC/{name_dataset}.npy`.
    - Se usa threading para paralelizar la descarga de batches.
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
    for batch in arr[1:]:
        imgs = np.load(f'{name_dataset}/{name_dataset}_{batch}.npy')
        full_imgs.append(imgs)

    full_imgs = np.concatenate(full_imgs, axis=0)
    np.save(f'data/SERSIC/{name_dataset}.npy', full_imgs.astype(np.float32))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe_path', type=str, default="data/SERSIC/df_coords_fix.csv", help='Ruta al dataframe')
    parser.add_argument('--img_size', type=int, default=30, help='Tamaño de las imagenes a descargar')
    parser.add_argument('--filters', type=str, default='grizy', help='Filtros a descargar')
    parser.add_argument('--n_procesos', type=int, default=16, help='Numero de hilos')
    parser.add_argument('--name_dataset', type=str, default="delight_multires_grizy", help='Nombre del dataset')

    args = parser.parse_args()

    df = pd.read_csv(args.dataframe_path)

    alpha = time.time()

    download_all(df, args.img_size, args.name_dataset, args.n_procesos, args.filters)

    shutil.rmtree(args.name_dataset)

    print(f"Fin Total: {time.time()-alpha} [s]")