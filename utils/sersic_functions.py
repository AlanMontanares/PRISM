import numpy as np

def sersic_profile(image_shape, x_center, y_center,
                    Re_arcsec, b_over_a, theta_deg,
                    pixel_scale=0.25, Ie=1.0, n=4):

    
    Re_pix = Re_arcsec / pixel_scale
    q = b_over_a
    theta_rad = np.deg2rad(theta_deg)
    b_n = 2 * n - 1/3 

    y, x = np.indices(image_shape)
    x_shifted = x - x_center
    y_shifted = y - y_center

    # Rotar coordenadas
    x_rot = x_shifted * np.cos(theta_rad) + y_shifted * np.sin(theta_rad)
    y_rot = -x_shifted * np.sin(theta_rad) + y_shifted * np.cos(theta_rad)

    # Radio elíptico
    r_ell = np.sqrt((x_rot / Re_pix)**2 + (y_rot / (Re_pix*q))**2) #re = a:  major axis , re*q = re*b/a = b: minor axis

    # Perfil Sérsic
    intensity = Ie * np.exp(-b_n * ((r_ell / Re_pix)**(1/n) - 1))

    # Máscara elíptica (dentro de la elipse intensidad > 0)
    ellipse_r = (x_rot / Re_pix)**2 + (y_rot / (Re_pix * q))**2
    mask = (ellipse_r <= 9)
    
    # Aplicar máscara: fuera de la elipse intensidad = 0
    intensity_masked = np.zeros_like(intensity, dtype=np.float32)
    intensity_masked[mask] = intensity[mask].astype(np.float32)

    return np.flipud(intensity_masked) # flip eje horizontal


def generate_random_pos(sersic_radius, sersic_ab, sersic_phi, img_size):

    x_center, y_center = img_size//2 -1, img_size//2 -1

    # Parámetros de Sérsic
    pixel_scale = 0.25

    # Generar perfil de Sérsic
    sersic_img = sersic_profile(
        image_shape=(img_size,img_size),
        x_center=x_center, y_center=y_center,
        Re_arcsec=sersic_radius,
        b_over_a=sersic_ab,
        theta_deg=sersic_phi,  
        pixel_scale=pixel_scale,
        Ie=1.0,
        n=4
    )

    pesos = sersic_img.ravel()
    pesos = pesos / pesos.sum()

    indice_aleatorio = np.random.choice(len(pesos), p=pesos)

    x_supernova, y_supernova = np.unravel_index(indice_aleatorio, sersic_img.shape) + np.random.uniform(-0.49999, 0.49999, size= 2) # Hacemos que la posicion este arbitrariamente dentro de ese pixel

    return np.array([x_supernova-x_center, y_supernova-y_center])