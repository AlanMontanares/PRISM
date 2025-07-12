import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
import time
import shutil

from astroquery.hips2fits import hips2fits
import argparse
import sys 
import os

from astropy.wcs import WCS

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from utils.sersic_functions import generate_random_pos


def get_wcs(ra, dec):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [15.0, 15.0]
    wcs.wcs.cdelt = [-6.944444445183e-5, 6.944444445183e-5]  # grados/píxel
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs

def get_galaxy_img(ra, dec, level, size):
    
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

def get_multires(df, idx, size, num_augmentations):
    '''
    Retorna la tupla (imgs, pos_host)

    imgs: imagenes en multi resolucion centradas en la SN
    pos_host: distancia desde el centro a la galaxia host en pixeles, en la forma (x,y)
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

        # Se genera una posicion de SN centrada en el host
        pos = generate_random_pos(
            sersic_radius=radius_sersic,
            sersic_ab=ab_sersic,
            sersic_phi=phi_sersic,
            img_size=600 # El radio maximo era 300 pix
        )

        wcs = get_wcs(host_ra, host_dec)
        
        # Le sumamos el centro a la posicion de la SN y obtenemos sus coordenadas (ra,dec)
        ra_sn, dec_sn = wcs.pixel_to_world_values([pos + 14])[0]

        # Obtenemos la imagen en multi-resolucion
        multi = []
        for i in range(5):
            img = get_galaxy_img(ra_sn, dec_sn, level=i, size=size)
            multi.append(img)

        imagenes.append(np.array(multi))
        posiciones.append(-pos) # Ahora la imagen esta centrada en la SN por lo que la posicion al host es lo opuesto

    return np.stack(imagenes) , np.stack(posiciones)

def download_batch(data_frame, inicio, final, name_dataset, num_augmentations):

    img_stack = []
    pos_stack = []

    max_retry = 2
    size = 30
    for x in tqdm(range(inicio,final)):

        for retry in range(max_retry):
            try:
                img, pos = get_multires(data_frame, x, size=size, num_augmentations= num_augmentations) 
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe_path', type=str, default="..\data\SERSIC\df_train.csv", help='Path of train metadata')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size to download imgs')
    parser.add_argument('--name_dataset', type=str, default="augmented_x1000", help='Name dataset')
    parser.add_argument('--num_augmentations', type=int, default=1, help='Numbers of augmentation per example')

    args = parser.parse_args()

    df = pd.read_csv(args.dataframe_path)

    alpha = time.time()

    t = threading.Thread(target=download_all, args=[df, args.name_dataset, args.batch_size, args.num_augmentations])
    t.start()
    t.join()

    shutil.rmtree(args.name_dataset)

    print(f"Fin Total: {time.time()-alpha} [s]")