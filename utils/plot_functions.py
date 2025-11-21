import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.wcs import WCS
import matplotlib.cm as cm
import re


def load__preds(name_run):
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


def print_metrics(results, name):
    """
    Imprime métricas de error para un experimento dado.

    Calcula RMSE, desviación media, mediana y moda de las distancias 
    residuales entre predicciones y targets (escaladas a arcsec).

    Parameters
    ----------
    results : dict
        Diccionario de resultados donde cada clave es el nombre de un experimento y 
        cada valor es una tupla (preds, targets, mean_preds, original_target).
    name : str
        Nombre del experimento cuyas métricas se desean imprimir.

    Returns
    -------
    None
        Imprime las métricas en consola.
    """
    preds = results[name][0]*0.25
    targets = results[name][1]*0.25
    mse = torch.nn.MSELoss()

    rmse = mse(preds,targets).sqrt()
    mean_deviation = get_distances(preds,targets)[0].mean()
    median_deviation = get_distances(preds,targets)[0].median()
    mode_deviation = get_distances(preds,targets)[0].mode().values

    print(f"RMSE: {rmse}")
    print(f"mean deviation: {mean_deviation}")
    print(f"median deviation: {median_deviation}")
    print(f"mode deviation: {mode_deviation}")


def plot_binned_residuals_vs_host_dist_grouped(resultados, experiments, titles, cmap_name,
                                               dpi=100, n_bins=5, metric="bias"):
    """
    metric: "bias" | "nmad"
        - bias = promedio ± std como antes
        - nmad = usa NMAD por bin
    """

    def compute_nmad(values):
        """NMAD = 1.4826 * median(|x - median(x)|)"""
        if len(values) == 0:
            return np.nan
        med = np.median(values)
        return 1.4826 * np.median(np.abs(values - med))

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
                        # igual que antes: mediana por run
                        stat = np.nanmedian(vals)

                    elif metric == "nmad":
                        # usar NMAD por run
                        stat = compute_nmad(vals)

                    else:
                        raise ValueError("metric debe ser 'bias' o 'nmad'")

                    bin_stats.append(stat)

                if len(bin_stats) > 0:
                    mean_vals[b] = np.nanmedian(bin_stats)
                    std_vals[b] = np.nanstd(bin_stats)

            valid = ~np.isnan(mean_vals)

            # Línea mean
            ax.plot(bin_centers[valid], mean_vals[valid], "-o",
                    lw=1.7, color=color, label=titles[idx])

            # Sombreado std
            ax.fill_between(bin_centers[valid],
                            mean_vals[valid] - std_vals[valid],
                            mean_vals[valid] + std_vals[valid],
                            color=color, alpha=0.25)

        # Config estilo plot
        ax.set_xlim(0, 70)
        if normalize_error:
            ax.set_title(f"Normalized {metric.upper()}")
            ax.set_ylabel(f"{metric.upper()} / Host Dist")
        else:
            ax.set_title(f"Absolute {metric.upper()}")
            ax.set_ylabel(f"{metric.upper()} ['']")
        ax.set_xlabel("Host Dist ['']")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_residuals_vs_host_dist_grouped(
    resultados, experiments, titles, cmap_name, masks=None, dpi=100
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

        axs[g_idx].set_xlabel("Host Dist ['']")
        axs[g_idx].set_aspect("equal")

        axs[g_idx].text(5, 55, titles[g_idx], color=color, fontsize=12)

        if g_idx != 0:
            axs[g_idx].set_yticklabels([])

    axs[0].set_ylabel("Residual Dist ['']")

    plt.tight_layout()
    plt.show()