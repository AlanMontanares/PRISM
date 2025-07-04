import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
import time
import shutil

import astropy.units as u
from astropy.coordinates import Longitude, Latitude, Angle

from astroquery.hips2fits import hips2fits
import argparse
import sys 
import os


from astropy.io.fits import Header
from astropy.wcs import WCS
import textwrap

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from utils.sersic_functions import generate_random_pos

def get_wcs(ra, dec):
        
    header_str = textwrap.dedent(f"""\
        SIMPLE  =                    T / conforms to FITS standard
        BITPIX  =                  -32 / array data type
        NAXIS   =                    2 / number of array dimensions
        NAXIS1  =                   30
        NAXIS2  =                   30
        WCSAXES =                    2 / Number of coordinate axes
        CRPIX1  =                 15.0 / Pixel coordinate of reference point
        CRPIX2  =                 15.0 / Pixel coordinate of reference point
        CDELT1  =  -6.944444445183E-05 / [deg] Coordinate increment at reference point
        CDELT2  =   6.944444445183E-05 / [deg] Coordinate increment at reference point
        CUNIT1  = 'deg'                / Units of coordinate increment and value
        CUNIT2  = 'deg'                / Units of coordinate increment and value
        CTYPE1  = 'RA---TAN'           / Right ascension, gnomonic projection
        CTYPE2  = 'DEC--TAN'           / Declination, gnomonic projection
        CRVAL1  =          {ra} / [deg] Coordinate value at reference point
        CRVAL2  =          {dec} / [deg] Coordinate value at reference point
        LONPOLE =                180.0 / [deg] Native longitude of celestial pole
        LATPOLE =          {dec} / [deg] Native latitude of celestial pole
        RADESYS = 'ICRS'               / Equatorial coordinate system
        CPYRIGHT= 'PS1 Science Consortium - http://panstarrs.stsci.edu/'
        HISTORY Generated by CDS HiPS2FITS service - See https://alasky.cds.unistra.fr/h
        HISTORY ips-image-services/hips2fits for details
        HISTORY From HiPS CDS/P/PanSTARRS/DR1/r (PanSTARRS DR1 r)
        HISTORY HiPS created by Boch T. (CDS) - CNRS/Universite de Strasbourg
    """)
    header = Header.fromstring(header_str, sep='\n')
    wcs = WCS(header)

    return wcs

def get_wcs2(ra, dec):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [15.0, 15.0]
    wcs.wcs.cdelt = [-6.944444445183e-5, 6.944444445183e-5]  # grados/píxel
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs

def get_galaxy_img(ra, dec, fov, size):
    
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


def get_multires(df, idx, fov, size, num_augmentations):
    '''
    Retorna la tupla (imgs, pos_host)

    imgs: imagenes en multi resolucion centradas en la SN
    pos_host: distancia desde el centro a la galaxia host en pixeles, en la forma (y,x)
    '''
    row = df.iloc[idx]
    radius_sersic = row["rSerRadius"]
    ab_sersic = row["rSerAb"]
    phi_sersic = row["rSerPhi"]

    host_ra = row["host_ra"]
    host_dec = row["host_dec"]

    # Posición arbitraria

    imagenes = []
    posiciones = []
    for x in range(num_augmentations):
        pos = generate_random_pos(
            sersic_radius=radius_sersic,
            sersic_ab=ab_sersic,
            sersic_phi=phi_sersic,
            img_size=600 # El radio maximo era 300 pix
        )

        wcs = get_wcs2(host_ra, host_dec)

        ra_sn, dec_sn = wcs.pixel_to_world_values([pos[::-1]+14])[0]

        multi = []
        for i in range(5):
            img = get_galaxy_img(ra_sn, dec_sn, fov=fov*(2**i), size=size)
            multi.append(img)

        imagenes.append(np.array(multi))
        posiciones.append(-pos[::-1])

    return np.stack(imagenes) , np.stack(posiciones)

def download_batch(data_frame, inicio, final, name_dataset, num_augmentations):

    img_stack = []
    pos_stack = []

    max_retry = 2
    size = 30
    for x in tqdm(range(inicio,final)):

        for retry in range(max_retry):
            try:
                img, pos = get_multires(data_frame, x, fov = size*0.25/3600, size=size, num_augmentations= num_augmentations) 
                img_stack.append(img)
                pos_stack.append(pos)
                break

            except:
                if retry+1 == max_retry:
                    img_stack.append(np.zeros((num_augmentations, 5,size,size), dtype=np.float32))
                    pos_stack.append(np.zeros((num_augmentations, 2), dtype=np.float32))

    np.savez(f'{name_dataset}/{name_dataset}_{final}.npz', imgs=np.concatenate(img_stack), pos=np.concatenate(pos_stack))


def download_all(df, name_dataset, batch_size, num_augmentations):

    os.makedirs(name_dataset, exist_ok=True)

    start = 0
    total = len(df)

    arr = np.arange(start, total, batch_size)
    arr = np.append(arr, total)

    n_procesos = len(arr)-1
    threads = []


    for i in range(n_procesos):
        stop = min(arr[i]+batch_size, total)
        t = threading.Thread(target=download_batch, args=[df, arr[i], stop, name_dataset, num_augmentations])
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

    np.savez(f'..\data\SERSIC\X_train_{name_dataset}.npz', imgs=full_imgs, pos=full_pos)

    #df["aug_dx"] = full_pos[:,0]
    #df["aug_dy"] = full_pos[:,1]

    #np.save(f'..\data\h2f_ps1_multires_{name_dataset}.npy', full_imgs.astype(np.float32))
    #df.to_csv(f'..\data\h2f_ps1_multires_{name_dataset}.csv', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe_path', type=str, default="simple", help='..\data\df.csv')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size to download imgs')
    parser.add_argument('--name_dataset', type=str, default="simple_method", help='Name dataset')
    parser.add_argument('--num_augmentations', type=int, default=1, help='Numbers of augmentation per example')

    args = parser.parse_args()

    df = pd.read_csv(args.dataframe_path)

    alpha = time.time()

    t = threading.Thread(target=download_all, args=[df, args.name_dataset, args.batch_size, args.num_augmentations])
    t.start()
    t.join()

    shutil.rmtree(args.name_dataset)

    print(f"Fin Total: {time.time()-alpha} [s]")