import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
import time
import os
import shutil
import xarray as xr

import astropy.units as u
from astropy.coordinates import Longitude, Latitude, Angle

from astroquery.hips2fits import hips2fits
import argparse

def get_galaxy_img(df, id, fov, size):
    
    ra,dec = np.float32(df.iloc[id][["host_ra","host_dec"]].values)

    r = hips2fits.query(
        hips="CDS/P/PanSTARRS/DR1/r",
        width=size,
        height=size,
        ra=Longitude(ra * u.deg),
        dec=Latitude(dec * u.deg),
        fov=Angle(fov  * u.deg),
        projection="TAN",
        get_query_payload=False,
        format='fits',
        )


    r = r[0].data.byteswap().newbyteorder()
    r = np.nan_to_num(r, 0)

    r = np.rot90(r, k=1)
    r = np.flipud(r)
    return r


def get_multires_like_delight(df, id, fov, size):

    nlevels = 5
    data = get_galaxy_img(df, id, fov=fov, size=size)

    delta = int(data.shape[0]/2**nlevels)
    datah = np.zeros((nlevels, 2 * delta, 2 * delta))
    
    # iterate each level
    for exp in range(nlevels):
        factor = 2**exp
        a = xr.DataArray(data, dims=['x', 'y'])
        c = a.coarsen(x=factor, y=factor).median()
        center = int(c.shape[0]/2)
        image = c[center-delta: center+delta, center-delta: center+delta]
        datah[exp] = image

    del data

    return datah


def get_multires(df, id, fov, size):
 
    multi = []
    for i in range(5):
        img = get_galaxy_img(df, id, fov=fov/(2**(4-i)), size=size)
        multi.append(img)

    return np.array(multi)


def download_batch(data_frame, inicio, final, name_dataset):

    stack = []
    max_retry = 2

    for x in tqdm(range(inicio,final)):

        for retry in range(max_retry):
            try:
                img = get_multires(data_frame, x, fov = 4320*0.25/3600, size=270)
                stack.append(img)
                break

            except:
                if retry+1 == max_retry:
                    stack.append(np.zeros((5,270,270), dtype=np.float32))

    np.save(f'{name_dataset}/{name_dataset}_{final}.npy', np.stack(stack))


def download_all(df, name_dataset, batch_size):

    os.makedirs(name_dataset, exist_ok=True)

    start = 0
    total = len(df)

    arr = np.arange(start, total, batch_size)
    arr = np.append(arr, total)

    n_procesos = len(arr)-1
    threads = []

    ini = time.time()

    for i in range(n_procesos):
        stop = min(arr[i]+batch_size, total)
        t = threading.Thread(target=download_batch, args=[df, arr[i], stop, name_dataset])
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

    print(f"Dataset Finalizado en {time.time()-ini} [s]")

    full = []
    for batch in arr[1:n_procesos+1]:
        stack = np.load(f'{name_dataset}/{name_dataset}_{batch}.npy')
        full.append(stack)
        
    full_final = np.concatenate(full, axis=0)
    full_final = np.transpose(full_final, (0, 2, 3, 1))

    np.save(f'..\data\h2f_ps1_multires_{name_dataset}.npy', full_final.astype(np.float32))

    shutil.rmtree(name_dataset)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe_path', type=str, default="simple", help='..\data\df.csv')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size to download imgs')

    args = parser.parse_args()


    df = pd.read_csv("..\data\SERSIC\delight_sersic.csv")

    alpha = time.time()

    t = threading.Thread(target=download_all, args=[df, "simple_method", args.batch_size])
    t.start()
    t.join()

    print(f"Fin Total: {time.time()-alpha} [s]")