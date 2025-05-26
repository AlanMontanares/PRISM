# PRISM: Photometric Redshift and host Identification of Supernovae via Multi-task learning

## Instalación
Requerimientos:

- Python (>=3.11, <3.13)
- Poetry: Mire las [instrucciones](https://python-poetry.org/docs/#installing-with-pipx).

Para instalar las dependencias necesarias ejecute el comando:

```python
poetry install
```

## Entrenamiento

Primero es necesario que cree una carpeta en el directorio llamada `data`, en la cual debera contener un archivo `.npz` con las imagenes a utilizar en multi-resolucion, estas deberan tener esta forma `(N, W, H, L)` (#n imagenes, ancho, alto, niveles). Si tiene las coordenadas celestes de las galaxias y/o supernovas, puede descargar las imagenes asociadas utilizando el archivo `utils/download_multi_res_data.py`.

Para efectuar un entrenamiento puede hacerlo via terminal de esta forma:

```python
python .\train.py --train_dataset_type delight_autolabeling --epoch 40 --save_files ../resultados/autolabeling --oids_origin SERSIC
```

