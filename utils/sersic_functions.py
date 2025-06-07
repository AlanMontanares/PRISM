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
    mask = ellipse_r <= 9 # 3**2

    # Aplicar máscara: fuera de la elipse intensidad = 0
    intensity_masked = np.zeros_like(intensity)
    intensity_masked[mask] = intensity[mask]

    #DELIGHT hace estas transformaciones
    intensity_masked = np.rot90(intensity_masked, k=1)
    intensity_masked = np.flipud(intensity_masked)

    return intensity_masked


def generate_random_pos(sersic_radius, sersic_ab, sersic_phi):

    x_center, y_center = 134, 134

    # Parámetros de Sérsic
    pixel_scale = 0.25

    # Generar perfil de Sérsic
    sersic_img = sersic_profile(
        image_shape=(270,270),
        x_center=x_center, y_center=y_center,
        Re_arcsec=sersic_radius,
        b_over_a=sersic_ab,
        theta_deg=sersic_phi,  # sin +90 aquí
        pixel_scale=pixel_scale,
        Ie=1.0,
        n=4
    )

    pesos = sersic_img.ravel()
    pesos = pesos / pesos.sum()

    indice_aleatorio = np.random.choice(len(pesos), p=pesos)

    x_supernova, y_supernova = np.unravel_index(indice_aleatorio, sersic_img.shape)

    return np.array([x_supernova-134, y_supernova-134])