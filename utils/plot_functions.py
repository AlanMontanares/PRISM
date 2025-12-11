import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.wcs import WCS
import matplotlib.cm as cm
import re

from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

from matplotlib.cm import viridis, plasma, jet, ScalarMappable


from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy import units as u
from tqdm import tqdm
from joblib import Parallel, delayed
import time


def load_preds(name_run):
    """
    Carga predicciones y targets desde un archivo de resultados.

    Parameters
    ----------
    name_run : str
        Nombre del experimento o carpeta que contiene el archivo 
        `../resultados/{name_run}/test_results.npz`.

    Returns
    -------
    preds : torch.Tensor
        Predicciones de la red para cada transformación.
    targets : torch.Tensor
        Valores reales para cada transformación.
    mean_preds : torch.Tensor
        Predicciones promedio para cada ejemplo.
    original_target : torch.Tensor
        Valores reales sin transformación para cada ejemplo.
    """
    results = np.load(f"../resultados/{name_run}/test_results.npz")
    mean_preds = torch.tensor(results["mean_preds"])
    original_target = torch.tensor(results["original_target"])

    return mean_preds, original_target

def load_redshift_preds(name_run, special_preds=False):
    """
    Carga predicciones y targets desde un archivo de resultados.

    Parameters
    ----------
    name_run : str
        Nombre del experimento o carpeta que contiene el archivo 
        `../resultados/{name_run}/test_results.npz`.

    Returns
    -------
    preds : torch.Tensor
        Predicciones de la red para cada transformación.
    targets : torch.Tensor
        Valores reales para cada transformación.
    mean_preds : torch.Tensor
        Predicciones promedio para cada ejemplo.
    original_target : torch.Tensor
        Valores reales sin transformación para cada ejemplo.
    """
    if special_preds:
        results = np.load(f"../resultados/{name_run}/test_results_redshift_2.npz")
    else:
        results = np.load(f"../resultados/{name_run}/test_results_redshift.npz")
    z_phot = torch.tensor(results["zphot"])
    z_spec = torch.tensor(results["zspect"])

    return z_phot, z_spec

def get_mse_df_from_resultados(resultados):
    """
    Calcula el MSE por transformación para cada experimento en un diccionario de resultados.

    Parameters
    ----------
    resultados : dict
        Diccionario donde cada clave es el nombre de un experimento y cada 
        valor es una tupla (preds, targets, mean_preds, original_target).

    Returns
    -------
    pd.DataFrame
        DataFrame con los experimentos como filas y los MSE para cada 
        transformación como columnas.
    """
    mse = torch.nn.MSELoss()
    
    transformations = {
        0: "Original",
        1: "Rotation 90°",
        2: "Rotation 180°",
        3: "Rotation 270°",
        4: "Horizontal Flip",
        5: "HF + Rot 90°",
        6: "HF + Rot 180°",
        7: "HF + Rot 270°",
    }

    data = []

    for name, (preds, targets, _, _) in resultados.items():
        row = {}
        for pos, trans_name in transformations.items():
            row[trans_name] = mse(targets[:, pos, :], preds[:, pos, :]).item()
        row["Experimento"] = name
        data.append(row)

    df = pd.DataFrame(data)
    df = df.set_index("Experimento")
    return df


def get_distances(pred,target):
    """
    Calcula las distancias entre predicciones y targets en píxeles.

    Parameters
    ----------
    pred : torch.Tensor
        Tensor con las predicciones (x, y) de la posición de la galaxia host.
    target : torch.Tensor
        Tensor con las posiciones reales (x, y) de la galaxia host.

    Returns
    -------
    residual_dist_pix : torch.Tensor
        Distancia entre cada predicción y su valor real.
    host_dist_pix : torch.Tensor
        Distancia de cada supernova a su host (target respecto al origen).
    """
    residual_dist_pix = torch.norm(pred - target, dim=1)    
    host_dist_pix = torch.norm(target, dim=1)   

    return residual_dist_pix, host_dist_pix


def plot_residuals_vs_host_dist(resultados, experiments, titles, cmap_name, masks=None, dpi=100):
    """
    Grafica la distancia residual vs. la distancia al host para varios experimentos.

    Parameters
    ----------
    resultados : dict
        Diccionario de resultados donde cada clave es el nombre de un experimento y cada 
        valor es una tupla (preds, targets, mean_preds, original_target).
    experiments : list of str
        Lista con los nombres de los experimentos a graficar (claves en `resultados`).
    titles : list of str
        Títulos a mostrar en cada subplot, en el mismo orden que `experiments`.
    cmap_name : str
        Nombre del colormap a usar para asignar colores a los experimentos.
    masks : list of np.ndarray or None, optional
        Lista de máscaras booleanas indicando puntos incorrectos por experimento.
        Si se provee, los puntos marcados se resaltan en rojo.
    dpi : int, optional
        Resolución de la figura en puntos por pulgada.

    Returns
    -------
    None
        Muestra la figura en pantalla.
    """
    fig, axs = plt.subplots(1, len(experiments), figsize=(4*len(experiments), 4), dpi=dpi)

    n = len(experiments)
    color_vals = np.linspace(0.2, 0.8, n)
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(v) for v in color_vals]

    for i, key in enumerate(experiments):
        mean_preds, original_target = resultados[key]
        residual_dist_pix, host_dist_pix = get_distances(mean_preds, original_target)

        x = host_dist_pix * 0.25
        y = residual_dist_pix * 0.25

        # Dibujar todos los puntos correctamente predichos
        axs[i].scatter(x, y, color=colors[i], alpha=0.7, label='Correct')

        # Si hay máscara de errores, sobreponer puntos rojos con borde negro
        if masks is not None and masks[i] is not None:
            incorrect_mask = masks[i]
            axs[i].scatter(
                x[incorrect_mask], y[incorrect_mask],
                color='red', edgecolor='black', linewidth=0.5, alpha=0.9, label='Incorrect'
            )

        axs[i].set_xlim(-3, 65)
        axs[i].set_ylim(-3, 65)
        axs[i].plot([0, 60], [0, 60], linestyle='--', color='gray')

        axs[i].set_xlabel("Host Dist ['']")
        axs[i].set_aspect('equal')
        axs[i].text(5, 55, titles[i], color=colors[i], fontsize=12)

        # Quitar ejes Y de todos excepto el primero
        if i != 0:
            axs[i].set_yticklabels([])

    axs[0].set_ylabel("Residual Dist ['']")

    plt.tight_layout()
    plt.show()


def obtain_predicted_ra_dec(df_test, mean_preds):
    """
    Convierte predicciones de píxeles a coordenadas astronómicas (RA, Dec).

    Para cada ejemplo en el DataFrame de test, convierte las predicciones de
    posición (en píxeles) a coordenadas de mundo (RA, Dec) usando WCS.

    Parameters
    ----------
    df_test : pandas.DataFrame
        DataFrame con la información de test. Debe contener una columna "sn_coords" 
        con las coordenadas de la supernova en formato "(ra, dec)".
    mean_preds : array
        Predicciones promedio en pixeles (N,2).


    Returns
    -------
    np.ndarray
        Array de forma (N, 2) con las coordenadas (RA, Dec) predichas para cada ejemplo.
    """
    def get_wcs(ra, dec):
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [15.0, 15.0]
        wcs.wcs.cdelt = [-6.944444445183e-5, 6.944444445183e-5]  # grados/píxel
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.cunit = ["deg", "deg"]
        return wcs

    coordinates = []
    for idx in range(len(df_test)):
    
        match = re.search(r'\(([-+]?(?:\d*\.\d+|\d+\.?)),\s*([-+]?(?:\d*\.\d+|\d+\.?))\)', df_test.iloc[idx]["sn_coords"])
        ra = np.float64(match.group(1))
        dec = np.float64(match.group(2))

        wcs = get_wcs(ra, dec)

        ra_host, dec_host = wcs.pixel_to_world_values([mean_preds[idx] +14])[0] # pred = (x,y) = (dec, ra)

        coordinates.append([ra_host, dec_host])
    
    return np.array(coordinates)

def print_group_metrics(results, experiments, names):
    """
    Calcula métricas agregadas (mean y std de las deviations) para grupos de experimentos.

    Parameters
    ----------
    results : dict
        Diccionario donde cada clave es el nombre de un experimento y 
        cada valor es una tupla (preds, targets, mean_preds, original_target).

    experiments : list of list
        Lista donde cada elemento es una lista de nombres de experimentos
        cuyos resultados deben agruparse.

    names : list
        Lista de nombres para imprimir cada grupo.

    Returns
    -------
    None
        Imprime las métricas en pantalla.
    """

    def compute_nmad(values):
        """NMAD = 1.4826 * median(|x - median(x)|)"""
        if len(values) == 0:
            return np.nan
        med = np.median(values)
        return 1.4826 * np.median(np.abs(values - med))

    for group, name in zip(experiments, names):
        all_deviations = []

        for exp in group:
            preds = results[exp][0] * 0.25
            targets = results[exp][1] * 0.25

            residual_dist_pix, host_dist_pix = get_distances(preds, targets)

            deviations = residual_dist_pix/host_dist_pix
            all_deviations.append(deviations)

        all_deviations = torch.stack(all_deviations)

        # Calcular métricas
        mean_dev = all_deviations.mean(1)
        median_dev = all_deviations.median(1).values
        mode_dev = all_deviations.mode(1).values
        mad_dev = torch.tensor([compute_nmad(all_deviations[i]) for i in range(len(all_deviations))])
        
        print(f"\n===== {name} =====")
        print(f"mean deviation:   {mean_dev.mean():.3f} +- {mean_dev.std():.2f}")
        print(f"median deviation: {median_dev.mean():.3f} +- {median_dev.std():.2f}")
        print(f"mode deviation:   {mode_dev.mean():.3f} +- {mode_dev.std():.2f}")
        print(f"mad deviation:   {mad_dev.mean():.3f} +- {mad_dev.std():.2f}")

def plot_binned_residuals_vs_host_dist_grouped(resultados, experiments, titles, cmap_name,
                                               dpi=100, n_bins=5, metric="bias",
                                               use_fill=True, fontsize=10,
                                               markers=None, marker_size=7,
                                               alphas=None):
    """
    metric: "bias" | "nmad"
    - bias = promedio ± std como antes
    - nmad = usa NMAD por bin

    markers: lista con markers para cada experimento
    marker_size: tamaño de los markers (igual para todos)
    alphas: lista con alpha por experimento (misma longitud que experiments)
    """

    def compute_nmad(values):
        """NMAD = 1.4826 * median(|x - median(x)|)"""
        if len(values) == 0:
            return np.nan
        med = np.median(values)
        return 1.4826 * np.median(np.abs(values - med))

    # ----------------------------
    # Preparar markers
    # ----------------------------
    if markers is None:
        markers = ["o"] * len(experiments)
    else:
        if len(markers) != len(experiments):
            raise ValueError("markers debe tener la misma longitud que experiments")

    # ----------------------------
    # Preparar alphas
    # ----------------------------
    if alphas is None:
        alphas = [1.0] * len(experiments)
    else:
        if len(alphas) != len(experiments):
            raise ValueError("alphas debe tener la misma longitud que experiments")

    # ----------------------------
    # 1) Agrupar residuals y hosts
    # ----------------------------
    grouped_host = []
    grouped_residual = []

    for group in experiments:
        host_runs = []
        residual_runs = []

        for key in group:
            mean_preds, original_target = resultados[key]
            residual_pix, host_pix = get_distances(mean_preds, original_target)

            # Convertir a numpy
            if not isinstance(host_pix, np.ndarray):
                host_pix = host_pix.detach().cpu().numpy()
            if not isinstance(residual_pix, np.ndarray):
                residual_pix = residual_pix.detach().cpu().numpy()

            # Pasar a arcsec
            host_runs.append(host_pix * 0.25)
            residual_runs.append(residual_pix * 0.25)

        grouped_host.append(np.vstack(host_runs))
        grouped_residual.append(np.vstack(residual_runs))

    # ----------------------------
    # 2) Bins
    # ----------------------------
    bins = np.linspace(0, 70, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # ----------------------------
    # 3) Figura
    # ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi)
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(v) for v in np.linspace(0.2, 0.8, len(experiments))]

    # ----------------------------
    # 4) Plots
    # ----------------------------
    for normalize_error, ax in zip([False, True], axes):
        for idx, (host_runs, residual_runs) in enumerate(zip(grouped_host, grouped_residual)):
            color = colors[idx]
            marker = markers[idx]
            alpha = alphas[idx]
            n_runs, n_samples = host_runs.shape

            # Normalización si corresponde
            if normalize_error:
                with np.errstate(divide='ignore', invalid='ignore'):
                    residual_runs = np.where(host_runs > 0,
                                             residual_runs / host_runs,
                                             np.nan)

            mean_vals = np.full(n_bins, np.nan)
            std_vals = np.full(n_bins, np.nan)

            # ----------------------------
            # 5) Calcular bias o NMAD
            # ----------------------------
            for b in range(n_bins):
                if b < n_bins - 1:
                    mask = (host_runs >= bins[b]) & (host_runs < bins[b + 1])
                else:
                    mask = (host_runs >= bins[b]) & (host_runs <= bins[b + 1])

                bin_stats = []  # un valor por run

                for r in range(n_runs):
                    vals = residual_runs[r][mask[r]]
                    if vals.size == 0:
                        continue

                    if metric == "bias":
                        stat = np.nanmean(vals)
                    elif metric == "nmad":
                        stat = compute_nmad(vals)
                    else:
                        raise ValueError("metric debe ser 'bias' o 'nmad'")

                    bin_stats.append(stat)

                if len(bin_stats) > 0:
                    mean_vals[b] = np.nanmean(bin_stats)
                    std_vals[b] = np.nanstd(bin_stats)

            valid = ~np.isnan(mean_vals)

            # Línea mean
            ax.plot(bin_centers[valid], mean_vals[valid],
                    "-", lw=1.7, color=color, alpha=alpha)

            # Markers con borde negro y alpha
            ax.plot(bin_centers[valid], mean_vals[valid],
                    linestyle="None",
                    marker=marker,
                    markersize=marker_size,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    alpha=alpha,
                    label=titles[idx])

            # Sombreado o barras de error
            if use_fill:
                ax.fill_between(bin_centers[valid],
                                mean_vals[valid] - std_vals[valid],
                                mean_vals[valid] + std_vals[valid],
                                color=color, alpha=0.25 * alpha)
            else:
                ax.errorbar(bin_centers[valid], mean_vals[valid],
                            yerr=std_vals[valid],
                            fmt="none",
                            ecolor=color,
                            elinewidth=1.3,
                            capsize=3,
                            alpha=alpha)

        metric_symbol = f"$<\\Delta h>$" if metric == "bias" else f"$\\sigma_{{\\mathrm{{MAD}}}}$"

        ax.set_xlim(0, 70)
        if normalize_error:
            ax.set_ylabel(f"{metric_symbol}", fontsize=fontsize, math_fontfamily='cm')
        else:
            ax.set_ylabel(f"{metric_symbol} $['']$", fontsize=fontsize, math_fontfamily='cm')

        ax.set_xlabel(r"$\mathrm{Host\ Dist}\ ['']$", fontsize=fontsize, math_fontfamily='cm')
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_residuals_vs_host_dist_grouped(
    resultados, experiments, titles,fontsize, fontsize_title, cmap_name, masks=None, dpi=100
):
    """
    Similar a plot_residuals_vs_host_dist, pero ahora `experiments`
    es una lista de listas, donde cada sublista contiene varios runs
    cuyos residuals deben ser promediados.
    """
    n_groups = len(experiments)
    fig, axs = plt.subplots(1, n_groups, figsize=(4*n_groups, 4), dpi=dpi)

    if n_groups == 1:
        axs = [axs]  # asegurar iterabilidad

    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(v) for v in np.linspace(0.2, 0.8, n_groups)]

    for g_idx, group in enumerate(experiments):
        color = colors[g_idx]

        # --------------------------------------------------
        # 1) Recolectar residuals y hosts de cada run del grupo
        # --------------------------------------------------
        host_list = []
        residual_list = []

        for key in group:
            mean_preds, original_target = resultados[key]
            residual_dist_pix, host_dist_pix = get_distances(mean_preds, original_target)

            residual_list.append(residual_dist_pix * 0.25)
            host_list.append(host_dist_pix * 0.25)

        host_array = np.vstack(host_list)      # shape: (n_runs, N)
        residual_array = np.vstack(residual_list)

        # --------------------------------------------------
        # 2) Tomar host_dist del primer run (son iguales)
        # --------------------------------------------------
        x = host_array[0]

        # --------------------------------------------------
        # 3) Promediar residual_dist_pix entre los runs
        # --------------------------------------------------
        y_mean = np.nanmean(residual_array, axis=0)

        # --------------------------------------------------
        # 4) Combinar máscaras (OR) si se entregan
        # --------------------------------------------------
        if masks is not None:
            group_masks = []
            for key in group:
                group_masks.append(masks[key])
            incorrect_mask = np.any(np.vstack(group_masks), axis=0)
        else:
            incorrect_mask = None

        # --------------------------------------------------
        # 5) Graficar
        # --------------------------------------------------
        axs[g_idx].scatter(x, y_mean, color=color, alpha=0.7, label="Mean Correct")

        # puntos incorrectos en rojo
        if incorrect_mask is not None:
            axs[g_idx].scatter(
                x[incorrect_mask], y_mean[incorrect_mask],
                color="red", edgecolor="black", linewidth=0.5, alpha=0.9,
                label="Incorrect"
            )

        axs[g_idx].set_xlim(-3, 65)
        axs[g_idx].set_ylim(-3, 65)
        axs[g_idx].plot([0, 60], [0, 60], linestyle="--", color="gray")

        axs[g_idx].set_xlabel(r"$\mathrm{Host\ Dist}\ ['']$", fontsize=fontsize, math_fontfamily='cm')
        axs[g_idx].set_aspect("equal")

        axs[g_idx].text(5, 55, titles[g_idx], color=color, fontsize=fontsize_title, math_fontfamily='cm')

        if g_idx != 0:
            axs[g_idx].set_yticklabels([])

    axs[0].set_ylabel(r"$\mathrm{Residual\ Dist}\ ['']$", fontsize=fontsize, math_fontfamily='cm')

    plt.tight_layout()
    plt.show()


def regression_metrics(y_true,y_pred):
    residuals = (y_pred- y_true)/(1+ y_true)

    bias = residuals.mean()
    nmad = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    foutliers = (np.abs(residuals)>0.05).sum()/len(residuals)
    return bias, nmad, foutliers


def plot_regresion(zphot1,zphot2, zspec1,zspec2,name1,name2,cmap = "jet",dpi=700):

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi, sharey=True, gridspec_kw={'wspace': 0})


    font_size = 10
    vmax= 23


    metricas_1 = regression_metrics(zspec1,zphot1)

    metricas_2 = regression_metrics(zspec2,zphot2)




    _, _, _, img  =axs[0].hist2d(zspec1, zphot1, bins=250, range=[[0, 0.32], [0, 0.32]], cmap=cmap,cmin=1, vmax=vmax)
    _, _, _, img2  =axs[1].hist2d(zspec2, zphot2, bins=250, range=[[0, 0.32], [0, 0.32]], cmap=cmap, cmin=1, vmax=vmax)

    frecuencias1 = img.get_array().data
    frecuencias2 = img2.get_array().data


    frecuencias1[frecuencias1==0] = np.nan
    frecuencias2[frecuencias2==0] = np.nan


    img.set_array(frecuencias1)
    img2.set_array(frecuencias2)

    axs[0].text(0.035, 0.3, f'{name1}', fontsize=12, color='black', ha='left', va='top')
    axs[0].text(0.035, 0.27, f"$<\\Delta z> =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    axs[0].text(0.035, 0.25, f"$\\sigma_{{MAD}} =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    axs[0].text(0.035, 0.23, f"$\\eta =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    axs[0].text(0.09, 0.27, f"{metricas_1[0]:.5f}", fontsize=font_size, color='black')
    axs[0].text(0.076, 0.25, f"{metricas_1[1]:.5f}", fontsize=font_size, color='black')
    axs[0].text(0.055, 0.23, f"{metricas_1[2]*100:.2f}%", fontsize=font_size, color='black')


    axs[1].text(0.035, 0.3, f'{name2}', fontsize=12, color='black', ha='left', va='top')
    axs[1].text(0.035, 0.27, f"$<\\Delta z> =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    axs[1].text(0.035, 0.25, f"$\\sigma_{{MAD}} =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    axs[1].text(0.035, 0.23, f"$\\eta =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    axs[1].text(0.09, 0.27, f"{metricas_2[0]:.5f}", fontsize=font_size, color='black')
    axs[1].text(0.076, 0.25, f"{metricas_2[1]:.5f}", fontsize=font_size, color='black')
    axs[1].text(0.055, 0.23, f"{metricas_2[2]*100:.2f}%", fontsize=font_size, color='black')

    for ax in axs:

        ax.set_xlim(0, 0.32)
        ax.set_ylim(0, 0.32)
        ax.set_box_aspect(1)

        ax.tick_params(axis='both', which='major', length=6, bottom=True, top=True,left=True, right=True, direction='in')   
        ax.tick_params(axis='both', which='minor', length=3, left=True, right=True, bottom=True, top=True, direction='in')   

        ax.set_xticks(np.linspace(0.0,0.32,33), minor=True)
        ax.set_xticks(np.linspace(0.05,0.3,6), minor=False)

        ax.set_yticks(np.linspace(0.0,0.32,33), minor=True)
        ax.set_yticks(np.linspace(0.0,0.3,7), minor=False)

        ax.plot([0, 0.32], [0, 0.32], linestyle='-', color='black', linewidth=0.7)
        ax.plot([0.05, 0.32], [0, 0.257143], linestyle='--', color='red', alpha=0.7)
        ax.plot([0, 0.257143], [0.05, 0.32], linestyle='--', color='red', alpha=0.7)

        ax.set_xlabel("ZSPEC")


    axs[0].set_ylabel("ZPHOT")


    norm = Normalize(vmin=0.2, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', pad=0.03)
    cbar.set_label('DENSIDAD')

    plt.show()


def nearest_sdss_galaxy(ra, dec, radius_arcmin=0.5):
    pos = coords.SkyCoord(ra, dec, unit="deg")
    while True:
        try:
            result = SDSS.query_crossid(
                pos,
                radius=radius_arcmin * u.arcmin,
                #spectro=True,
                photoobj_fields=['objid'],
                #specobj_fields=['zWarning','targetType','survey','primtarget'],
                data_release=17,
                cache=False,
            )
            # --- Caso: coordenadas fuera del footprint (respuesta vacía, NO error HTTP) ---
            if result is None or len(result) == 0:
                return None   # ← devolver NaN/None inmediatamente

            # --- Filtro según tus condiciones ---
            # mask = (
            #     (result['zWarning'] == 0) &
            #     (result['targetType'] == 'SCIENCE') &
            #     (result['survey'] == 'sdss') &
            #     (result['primtarget'] >= 64)
            # )
            # result = result[mask]

            # Nada cumple las condiciones → también es un "no match"
            if len(result) == 0:
                return None

            # Si llegamos aquí: todo bien → retornar el más cercano
            return result[0], len(result)

        except Exception as e:
            #print(f"Error con RA={ra}, DEC={dec}: {e}")
            #SDSS.clear_cache()
            time.sleep(1)  
            continue

def _crossmatch_single_row(row, ra_col, dec_col, radius_arcmin):
    ra = float(row[ra_col])
    dec = float(row[dec_col])
    out = nearest_sdss_galaxy(ra, dec, radius_arcmin)

    if out is None:
        return {
            'sdss_objid': pd.NA,
            #'sdss_ra': None,
            #'sdss_dec': None,
            #'sdss_dist_arcsec': None,
        }
    else:
        (obj, matches) = out
        return {
            'sdss_objid': int(obj['objid']),
            'matches': int(matches),
            #'sdss_ra': float(obj['ra']),
            #'sdss_dec': float(obj['dec']),
            #'sdss_dist_arcsec': float(dist_arcsec),
        }

def sdss_crossmatch_joblib(df_path, ra_col='ra', dec_col='dec', radius_arcmin=0.5, n_jobs=-1):

    df = pd.read_csv(df_path)
    rows = list(df.to_dict("records"))

    # tqdm + joblib
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_crossmatch_single_row)(row, ra_col, dec_col, radius_arcmin)
        for row in tqdm(rows)
    )

    df_out = pd.DataFrame(results)
    df_out["sdss_objid"] = df_out["sdss_objid"].astype("Int64")

    return df_out


def match_objid(df_target, df_preds):
    preds = (df_preds['sdss_objid'] == df_target['sdss_objid'])

    n_total_objid = df_target['sdss_objid'].notna().sum()
    n_match_objid = preds.eq(True).sum()

    accuracy = n_match_objid/n_total_objid

    return n_match_objid, accuracy


def metrics_cross_objid(results, experiments, names, name_real_objid):
    for group, name in zip(experiments, names):
        all_matches = []
        all_accuracies = []

        for exp in group:
            
            n_match_objid, accuracy = match_objid(results[name_real_objid], results[exp])
            all_matches.append(n_match_objid)
            all_accuracies.append(accuracy)

        all_matches = np.array(all_matches)
        all_accuracies = np.array(all_accuracies)

        print(f"\n===== {name} =====")
        print(f"mean matches:   {all_matches.mean():.4f} +- {all_matches.std():.2f}")
        print(f"mean accuracy: {all_accuracies.mean():.4f} +- {all_accuracies.std():.4f}")


def plot_regresion_multi(zphots, zspecs, names, cmap="jet", dpi=700):
    """
    zphots: lista de arrays zphot
    zspecs: lista de arrays zspec
    names: lista de strings (nombres de cada experimento)
    """

    n = len(zphots)
    assert len(zspecs) == n and len(names) == n, "Las listas deben tener la misma longitud."

    fig, axs = plt.subplots(
        1, n,
        figsize=(6*n, 5),
        dpi=dpi,
        sharey=True,
        gridspec_kw={'wspace': 0}
    )

    if n == 1:
        axs = [axs]  # asegurar iterable

    font_size = 10
    vmax = 23

    # Prepara colorbar global
    norm = Normalize(vmin=0.2, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    for i in range(n):
        zphot = zphots[i]
        zspec = zspecs[i]
        name  = names[i]

        metricas = regression_metrics(zspec, zphot)

        # hist2d
        _, _, _, img = axs[i].hist2d(
            zspec, zphot,
            bins=250,
            range=[[0, 0.32], [0, 0.32]],
            cmap=cmap,
            cmin=1,
            vmax=vmax
        )

        # Cambiar ceros por NaN para colormap
        frec = img.get_array().data
        frec[frec == 0] = np.nan
        img.set_array(frec)

        # Texto
        axs[i].text(0.035, 0.29, f"{name}", fontsize=12, color='black')
        axs[i].text(0.035, 0.27, "$<\\Delta z> =$", fontsize=font_size, fontweight='bold')
        axs[i].text(0.035, 0.25, "$\\sigma_{MAD} =$", fontsize=font_size, fontweight='bold')
        axs[i].text(0.035, 0.23, "$\\eta =$", fontsize=font_size, fontweight='bold')

        axs[i].text(0.09,  0.27, f"{metricas[0]:.5f}", fontsize=font_size)
        axs[i].text(0.076, 0.25, f"{metricas[1]:.5f}", fontsize=font_size)
        axs[i].text(0.055, 0.23, f"{metricas[2]*100:.2f}%", fontsize=font_size)

        # Estética
        ax = axs[i]
        ax.set_xlim(0, 0.32)
        ax.set_ylim(0, 0.32)
        ax.set_box_aspect(1)

        # --- TICKS EN LOS 4 LADOS ---
        ax.tick_params(axis='both', which='major', length=6,
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='both', which='minor', length=3,
                       direction='in',
                       bottom=True, top=True, left=True, right=True)

        # --- UBICACIÓN DE TICKS ---
        ax.set_xticks(np.linspace(0.0, 0.32, 33), minor=True)
        ax.set_xticks(np.linspace(0.05, 0.3, 6), minor=False)

        ax.set_yticks(np.linspace(0.0, 0.32, 33), minor=True)
        ax.set_yticks(np.linspace(0.0, 0.3, 7), minor=False)

        # --- LÍNEAS ---
        ax.plot([0, 0.32], [0, 0.32], '-', color='black', linewidth=0.7)
        ax.plot([0.05, 0.32], [0, 0.257143], '-', color='red', linewidth=0.9, alpha=1)
        ax.plot([0, 0.257143], [0.05, 0.32], '-', color='red', linewidth=0.9,alpha=1)

        ax.set_xlabel("ZSPEC")

    axs[0].set_ylabel("ZPHOT")

    # Colorbar global
    fig.colorbar(sm, ax=axs, orientation='vertical', pad=0.03).set_label("DENSIDAD")

    plt.show()


def metric_bin(zphot,zspec,feature,n_bins=5, metrica =""):

    res = ((zphot - zspec)/(1 + zspec))

    intervalosz = np.linspace(0, 70, n_bins + 1)
    
    metricaz = np.zeros(len(intervalosz) - 1)

    for i in range(len(metricaz)):
        intervalo_inferior = intervalosz[i]
        intervalo_superior = intervalosz[i + 1]

        datos_intervalo = res[(feature >= intervalo_inferior) & (feature < intervalo_superior)]

        if metrica == "bias":
            metricaz[i] = (datos_intervalo).mean()

        elif metrica == "mad":
            metricaz[i] = 1.4826 * np.median(np.abs(datos_intervalo - np.median(datos_intervalo)))
        
    return metricaz, intervalosz


def plot_metrics_bins(z_spect, zphots_list, host_dist, names, alphas, fmts, capsizes, n_bins,
                      metric="bias", cmap_name="plasma", ax=None,
                      figsize=(7, 5), dpi=200, fontsize=14, ylim=0.03):

    cmap = cm.get_cmap(cmap_name, len(zphots_list))
    colors = cmap(np.linspace(0.8, 0, len(zphots_list)))

    # Crear figura/ejes
    if ax is None:
        fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        axs = ax
        fig = axs.figure

    # Eje derecho para bias o mad
    ax3 = axs.twinx()

    # Histograma de host_dist
    bines = np.linspace(0, 70, n_bins + 1)

    axs.hist(host_dist, bins=bines, color="gray",
             histtype="stepfilled", alpha=0.5, edgecolor="gray")

    # ----------- AHORA EL EJE X ES LINEAL -----------
    axs.set_xscale("linear")

    # Métricas por bin
    for x in range(len(zphots_list)):
        promedios = []
        intervalos_global = None

        # zphots_list[x] **no** es lista → ahora es un único array
        promedio, intervalos = metric_bin(
            zphots_list[x], z_spect,
            n_bins=n_bins,
            feature=host_dist,
            metrica=metric
        )
        promedios.append(promedio)
        intervalos_global = intervalos

        promedios = np.array(promedios)
        med_x = (intervalos_global[:-1] + intervalos_global[1:]) / 2

        ax3.errorbar(
            med_x,
            promedios.mean(axis=0),
            fmt=fmts[x],
            markeredgecolor='black',
            markersize=6,
            label=names[x],
            markeredgewidth=1,
            alpha=alphas[x],
            capsize=capsizes[x],
            color=colors[x]
        )

    # ----------- FORMATO EJES -----------

    # Eje derecho (bias o mad)
    ax3.yaxis.tick_left()
    ax3.yaxis.set_label_position('left')
    ax3.yaxis.set_ticks_position('left')

    if metric == "bias":
        ax3.set_ylabel("$<\\Delta z>$", fontsize=fontsize, math_fontfamily='cm')
        ax3.axhline(y=0, color='gray', linestyle='dotted')
        ax3.set_ylim([-ylim, ylim])
    else:
        ax3.set_ylabel("$\\sigma_{MAD}$", fontsize=fontsize, math_fontfamily='cm')

    # Eje izquierdo (histograma)
    axs.yaxis.tick_right()
    axs.yaxis.set_label_position('right')
    axs.yaxis.set_ticks_position('right')
    #axs.set_ylabel("N", fontsize=fontsize)
    axs.text(76, 500000, "N", fontsize=12, ha='left', va='bottom')

    # Escala del eje derecho también lineal
    ax3.set_xscale("linear")
    axs.set_ylim([0.1, 1000000000000])
    axs.set_yscale("log")

    minor_ticks = np.concatenate([
        np.arange(0.1, 1, 0.1),
        np.arange(1, 10, 1),
        np.arange(10, 100, 10),
        np.arange(100, 1000, 100),
        np.arange(1000, 10000, 1000),
        np.arange(10000, 100000, 10000)


    ])
    
    axs.set_yticks([1, 10, 100, 1000,10000,100000], minor=False)    
    axs.set_yticks(minor_ticks, minor=True)
    axs.yaxis.set_minor_formatter(plt.NullFormatter())



    axs.set_xlabel(r"$\mathrm{Host\ Dist}\ ['']$", fontsize=fontsize, math_fontfamily='cm')

    # ----------- LEYENDA DENTRO DE LA FIGURA -----------

    ax3.legend(
        loc="upper right",
        fontsize=10,
    )

    return fig, axs, ax3


def plot_residuals_vs_host_dist_grouped2(
    resultados, experiments, titles, cmap_name,fontsize, fontsize_title, masks=None, dpi=100
):
    """
    Similar a plot_residuals_vs_host_dist, pero ahora `experiments`
    es una lista de listas, donde cada sublista contiene varios runs
    cuyos residuals deben ser promediados.
    """

    import math

    n_groups = len(experiments)

    # -------------------------------
    # 1) Definir grid
    # -------------------------------
    if n_groups == 6:
        rows, cols = 2, 3
    else:
        rows = 1
        cols = n_groups

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=dpi)
    axs = np.array(axs).reshape(-1)  # aplanar para indexar linealmente

    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(v) for v in np.linspace(0.2, 0.8, n_groups)]

    for g_idx, group in enumerate(experiments):
        ax = axs[g_idx]
        color = colors[g_idx]

        # --------------------------------------------------
        # 1) Recolectar residuals y hosts de cada run del grupo
        # --------------------------------------------------
        host_list = []
        residual_list = []

        for key in group:
            mean_preds, original_target = resultados[key]
            residual_dist_pix, host_dist_pix = get_distances(mean_preds, original_target)

            residual_list.append(residual_dist_pix * 0.25)
            host_list.append(host_dist_pix * 0.25)

        host_array = np.vstack(host_list)
        residual_array = np.vstack(residual_list)

        x = host_array[0]
        y_mean = np.nanmean(residual_array, axis=0)

        # --------------------------------------------------
        # Combinar máscaras (OR)
        # --------------------------------------------------
        if masks is not None:
            group_masks = [masks[key] for key in group]
            incorrect_mask = np.any(np.vstack(group_masks), axis=0)
        else:
            incorrect_mask = None

        # --------------------------------------------------
        # Graficar
        # --------------------------------------------------
        ax.scatter(x, y_mean, color=color, alpha=0.7, label="Mean Correct")

        if incorrect_mask is not None:
            ax.scatter(
                x[incorrect_mask], y_mean[incorrect_mask],
                color="red", edgecolor="black", linewidth=0.5, alpha=0.9,
                label="Incorrect"
            )

        ax.set_xlim(-3, 65)
        ax.set_ylim(-3, 65)
        ax.plot([0, 60], [0, 60], linestyle="--", color="gray")

        ax.set_xlabel(r"$\mathrm{Host\ Dist}\ ['']$", fontsize=fontsize, math_fontfamily='cm')
        ax.set_aspect("equal")

        ax.text(5, 55, titles[g_idx], color=color, fontsize=fontsize_title, math_fontfamily='cm')

        # solo mostrar eje Y en la primera columna
        if g_idx % cols != 0:
            ax.set_yticklabels([])

    axs[0].set_ylabel(r"$\mathrm{Residual\ Dist}\ ['']$", fontsize=fontsize, math_fontfamily='cm')
    axs[3].set_ylabel(r"$\mathrm{Residual\ Dist}\ ['']$", fontsize=fontsize, math_fontfamily='cm')

    plt.tight_layout()
    plt.show()