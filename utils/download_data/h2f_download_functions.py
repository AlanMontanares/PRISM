
import numpy as np

from astroquery.hips2fits import hips2fits
from astropy.wcs import WCS


from utils.sersic_functions import generate_random_pos

def get_wcs(ra, dec, center):
    """
    Crea un sistema de coordenadas WCS (World Coordinate System) centrado en un punto dado.

    Parameters
    ----------
    ra : float
        Ascensión recta del centro en grados.
    dec : float
        Declinación del centro en grados.
    center: float
        Centro de la imagen

    Returns
    -------
    astropy.wcs.WCS
        Objeto WCS configurado con proyección tangencial (TAN), 
        escala de 0.25 arcsec/pix (6.9444e-5 deg/pix) y centro en (ra, dec).
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [center, center]
    wcs.wcs.cdelt = [-6.944444445183e-5, 6.944444445183e-5]  # grados/píxel
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs

def get_galaxy_img(df, id, level, size, filters='r'):

    """
    Descarga una imagen astronómica del catálogo Pan-STARRS usando hips2fits 
    y un sistema de coordenadas WCS definido a partir de un objeto en el DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene al menos las columnas 'host_ra' y 'host_dec'
        con las coordenadas de los objetos en grados.
    id : int
        Índice (fila) del objeto en el DataFrame para el cual se descargará la imagen.
    level : int
        Nivel resolución de la imagen. A mayor nivel, menor resolución.
    size : int
        Tamaño de la imagen en píxeles (ancho y alto).

    Returns
    -------
    np.ndarray
        Imagen en formato numpy array 2D, sin valores NaN (estos se reemplazan por 0).
    """

    ra,dec = np.float64(df.iloc[id][["host_ra","host_dec"]].values)

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

    imgs = []

    for filter in list(filters):
        result = hips2fits.query_with_wcs(
            hips=f'CDS/P/PanSTARRS/DR1/{filter}',
            wcs=w,
            get_query_payload=False,
            format='fits')

        r = result[0].data.byteswap().newbyteorder()
        r = np.nan_to_num(r, 0)
        imgs.append(r)

    return r


def get_sn_img(sn_ra, sn_dec, level, size, filters='r'):

    """
    Descarga una imagen astronómica del catálogo Pan-STARRS usando hips2fits 
    y un sistema de coordenadas WCS definido a partir de un objeto en el DataFrame.

    Parameters
    ----------
    sn_ra : float
        Ra de la supernova en grados.
    sn_dec : float
        Dec de la supernova en grados.
    level : int
        Nivel resolución de la imagen. A mayor nivel, menor resolución.
    size : int
        Tamaño de la imagen en píxeles (ancho y alto).

    Returns
    -------
    np.ndarray
        Imagen en formato numpy array 3D (channel, alto, ancho), sin valores NaN (estos se reemplazan por 0).
    """

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
        'CRVAL1': sn_ra,
        'CRVAL2': sn_dec,
    })

    imgs = []

    for filter in list(filters):
        result = hips2fits.query_with_wcs(
            hips=f'CDS/P/PanSTARRS/DR1/{filter}',
            wcs=w,
            get_query_payload=False,
            format='fits')

        r = result[0].data.byteswap().newbyteorder()
        r = np.nan_to_num(r, 0)
        imgs.append(r)

    return np.array(imgs)


def get_multires(df, id, size, filters):
    """
    Genera un conjunto de imágenes multiresolución de una galaxia.

    Para una galaxia específica del DataFrame, descarga imágenes a diferentes
    niveles de resolución (level=0..4) manteniendo el mismo centro y tamaño.
    Cada una tiene la mitad de resolución y el doble de campo de vision que la anterior.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con la información de las galaxias. Debe contener al menos 
        'host_ra' y 'host_dec', que son utilizados internamente por `get_galaxy_img`.
    id : int
        Índice de la galaxia en el DataFrame.
    size : int
        Tamaño (en píxeles) de cada imagen generada.

    Returns
    -------
    np.ndarray
        Arreglo de forma (5, channels, size, size) que contiene las imágenes de la galaxia 
        en diferentes resoluciones (level=0..4).
    """

    #ra, dec = df[['sn_ra', 'sn_dec']].iloc[id].values
    ra, dec = df[['host_ra', 'host_dec']].iloc[id].values

    multi = []
    for i in range(5):
        img = get_sn_img(ra, dec, level=i, size=size, filters=filters)
        multi.append(img)

    return np.array(multi)


def get_augmented_multires(df, idx, size, num_augmentations, filters):
    """
    Toma un ejemplo de galaxia y genera distintas imagenes centradas en SN generadas segun su perfil de Sérsic,
    para cada imagen tambien se generan sub-imagenes en diferentes resoluciones y campos de vision,
    cada una tiene la mitad de resolución y el doble de campo de vision que la anterior.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con parámetros de las galaxias. Debe contener:
        - 'rSerRadius' : radio efectivo en arcsec
        - 'rSerAb' : relación de ejes b/a
        - 'rSerPhi' : ángulo de posición (grados)
        - 'host_ra', 'host_dec' : coordenadas del centro en grados
    idx : int
        Índice de la galaxia en el DataFrame.
    size : int
        Tamaño (en píxeles) de cada imagen de salida.
    num_augmentations : int
        Número de ejemplos de data augmentation a generar (posiciones aleatorias de SN).
    filters : str
        Filtros a descargar (grizy)

    Returns
    -------
    imagenes : np.ndarray
        Array de forma (num_augmentations, 5, size, size) que contiene
        imágenes multiresolución de la galaxia centradas en la posición de la SN.
        La dimensión 5 corresponde a los niveles de resolución o campo de visión (level=0..4).
    posiciones : np.ndarray
        Arreglo de forma (num_augmentations, 2) con las posiciones relativas
        del host respecto a la SN en píxeles: [dx, dy].

    Notes
    -----
    - Se usa `generate_random_pos` para muestrear posiciones de SN dentro del perfil de Sérsic.
    - Las imágenes multiresolución se generan con `get_galaxy_img`.
    - `pos` es el vector de galaxia a SN, por lo que `-pos` es el vector opuesto (target).
    """
    row = df.iloc[idx]
    radius_sersic = row["rSerRadius"]
    ab_sersic = row["rSerAb"]
    phi_sersic = row["rSerPhi"]

    host_ra = row["host_ra"]
    host_dec = row["host_dec"]


    imagenes = []
    posiciones = []
    for _ in range(num_augmentations):

        # Se genera una posicion de SN centrada en el host
        pos = generate_random_pos(
            sersic_radius=radius_sersic,
            sersic_ab=ab_sersic,
            sersic_phi=phi_sersic,
            img_size=600 # El radio maximo era 300 pix
        )

        wcs = get_wcs(host_ra, host_dec, size/2)
        
        # Le sumamos el centro a la posicion de la SN y obtenemos sus coordenadas (ra,dec)
        ra_sn, dec_sn = wcs.pixel_to_world_values([pos + size//2 -1])[0]

        # Obtenemos la imagen en multi-resolucion
        multi = []
        for i in range(5):
            img = get_sn_img(ra_sn, dec_sn, level=i, size=size, filters=filters)
            multi.append(img)

        imagenes.append(np.array(multi))
        posiciones.append(-pos) # Ahora la imagen esta centrada en la SN por lo que la posicion al host es lo opuesto

    return np.stack(imagenes) , np.stack(posiciones)