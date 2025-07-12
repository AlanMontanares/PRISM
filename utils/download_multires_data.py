import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
import time
import os
import shutil
#import xarray as xr

import astropy.units as u
from astropy.wcs import WCS

from astroquery.hips2fits import hips2fits
import argparse
import re

def get_galaxy_img(df, id, level, size):
    
    match = re.search(r'\(([-+]?(?:\d*\.\d+|\d+\.?)),\s*([-+]?(?:\d*\.\d+|\d+\.?))\)', df.iloc[id]["sn_coords"])
    ra = np.float64(match.group(1))
    dec = np.float64(match.group(2))

    w = WCS(header={
        'NAXIS': 2,
        'NAXIS1': size,
        'NAXIS2': size,
        'CTYPE1': 'RA---TAN',
        'CTYPE2': 'DEC--TAN',
        'CDELT1': -6.94444461259988E-05 * (2 ** level),  
        'CDELT2': 6.94444461259988E-05 * (2 ** level),  
        'CRPIX1': size/2,
        'CRPIX2': size/2,
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'CRVAL1': ra,
        'CRVAL2': dec,
    })

    result = hips2fits.query_with_wcs(
        hips='CDS/P/PanSTARRS/DR1/r',
        wcs=w,
        get_query_payload=False,
        format='fits')


    r = result[0].data.byteswap().newbyteorder()
    r = np.nan_to_num(r, 0)

    return r

# def get_multires_like_delight(df, id, fov, size):

#     nlevels = 5
#     data = get_galaxy_img(df, id, fov=fov, size=size)

#     delta = int(data.shape[0]/2**nlevels)
#     datah = np.zeros((nlevels, 2 * delta, 2 * delta))
    
#     # iterate each level
#     for exp in range(nlevels):
#         factor = 2**exp
#         a = xr.DataArray(data, dims=['x', 'y'])
#         c = a.coarsen(x=factor, y=factor).median()
#         center = int(c.shape[0]/2)
#         image = c[center-delta: center+delta, center-delta: center+delta]
#         datah[exp] = image

#     del data

#     return datah


def get_multires(df, id, size):
 
    multi = []
    for i in range(5):
        img = get_galaxy_img(df, id, level=i, size=size)
        multi.append(img)

    return np.array(multi)


def download_batch(data_frame, inicio, final, name_dataset):

    stack = []
    max_retry = 2
    size = 30

    for x in tqdm(range(inicio,final)):

        for retry in range(max_retry):
            try:
                img = get_multires(data_frame, x, size=size)  
                #img = get_multires_like_delight(data_frame, x, fov = size*0.25/3600, size=size)  
                stack.append(img)
                break

            except:
                if retry+1 == max_retry:
                    stack.append(np.zeros((5,size,size), dtype=np.float32))

    np.save(f'{name_dataset}/{name_dataset}_{final}.npy', np.stack(stack))


def download_all(df, name_dataset, batch_size):

    os.makedirs(name_dataset, exist_ok=True)

    start = 0
    total = len(df)

    arr = np.arange(start, total, batch_size)
    arr = np.append(arr, total)

    n_procesos = len(arr)-1
    threads = []


    for i in range(n_procesos):
        stop = min(arr[i]+batch_size, total)
        t = threading.Thread(target=download_batch, args=[df, arr[i], stop, name_dataset])
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()


    full = []
    for batch in arr[1:n_procesos+1]:
        stack = np.load(f'{name_dataset}/{name_dataset}_{batch}.npy')
        full.append(stack)
        
    full_final = np.concatenate(full, axis=0)
    full_final = np.transpose(full_final, (0, 2, 3, 1))

    np.save(f'..\data\SERSIC\{name_dataset}.npy', full_final.astype(np.float32))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe_path', type=str, default="..\data\SERSIC\df.csv", help='df train path')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size to download imgs')
    parser.add_argument('--name_dataset', type=str, default="dataset", help='Name dataset')

    args = parser.parse_args()


    df = pd.read_csv(args.dataframe_path)

    alpha = time.time()

    t = threading.Thread(target=download_all, args=[df, args.name_dataset, args.batch_size])
    t.start()
    t.join()

    shutil.rmtree(args.name_dataset)

    print(f"Fin Total: {time.time()-alpha} [s]")