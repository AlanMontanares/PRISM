# PRISM: Photometric Redshift and host Identification of Supernovae via Multi-task learning

Resumen
-------
PRISM es una implementación de aprendizaje multi-tarea para estimar redshifts fotométricos y asignar galaxias anfitrionas a supernovas usando imágenes multi-resolución. Está diseñada para investigación y experimentos reproducibles sobre conjuntos de datos astronómicos.

Características
---------------
- Entrenamiento multi-tarea (redshift + identificación de host).
- Soporte para imágenes multi-resolución en formato numpy (.npz).
- Scripts utilitarios para descargar/procesar datos.
- Compatible con CPU y GPU (PyTorch).

Requerimientos
--------------
- Python >= 3.11, < 3.13
- Poetry (recomendado) — ver https://python-poetry.org/docs/#installing-with-pipx

Para instalar las dependencias necesarias ejecute el comando:

```python
poetry install
```

# Estructura de datos

1. Cree un directorio `data/` en la raíz del proyecto.  
2. El archivo de imágenes debe ser un `.npz` con un array de forma `(N, W, H, L)`:
   - **N**: número de imágenes  
   - **W**: ancho  
   - **H**: alto  
   - **L**: niveles / canales multi-resolución  

Si dispone de coordenadas celestes (**RA/DEC**) para galaxias y/o supernovas, puede descargar imágenes asociadas utilizando:

```bash
python utils/download_multi_res_data.py

---

# 🧠 Entrenamiento

### Ejemplo básico (Linux)

```bash
python ./train.py --train_dataset_type delight_autolabeling --epoch 40 --save_files ./resultados/autolabeling --oids_origin SERSIC
