# watermarking

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

El proyecto al ser de investigación actualmente tiene un avance previo que este será el punto de partida donde se cuenta con la prueba de concepto, una IA capaz de generar marcas de agua y un modelo detector que determina si una firma contiene o no una marca de agua, el siguiente avance en el proyecto es generar marcas de agua basadas en una estampa de tiempo donde la IA tendrá como salida esperada la fecha en que fue escrita la firma, de igual manera se realizará la mejora de escalabilidad del proyecto para hacerlo más robusto para futuros avances por parte de otros equipos.

## Características:

**Gestión de datos:** El proyecto incluye scripts para gestionar y versionar los datos utilizados para entrenar el modelo.<br>

**Preprocesamiento de datos:** El proyecto incluye scripts para limpiar, transformar y preparar los datos para el entrenamiento del modelo. <br>

**Entrenamiento de modelos:** El proyecto incluye scripts para entrenar un modelo de aprendizaje automático utilizando los datos preparados.<br>

**Evaluación de modelos:** El proyecto incluye scripts para evaluar el rendimiento del modelo entrenado.<br>

**Despliegue de modelos:** El proyecto incluye scripts para desplegar el modelo entrenado en un entorno de producción.<br>

**Automatización:** El proyecto incluye scripts para automatizar todo el flujo de trabajo, desde la gestión de datos hasta el despliegue de modelos.<br>

## Configuración del Proyecto

### Configuración para uso de notebooks

Clona el repositorio:
   ```
   git clone https://github.com/isak792/watermarking
   cd watermark 
   ```
 
 1. Una vez instalados todos los requerimientos, seleccionamos la carpeta de notebooks

 2. Descargamos el archivo de la data con extension .pkl del siguiente link: 

 ```
https://drive.google.com/file/d/1RhhR9uVzSvyixuMQNrJ8FYFiXPA26zyu/view?usp=sharing

```
3. Generamos una carpeta llamada "data" y dentro de esta otra llamada "raw" en el directorio donde clonamos el proyecto.

```
 Bash

     mkdir -p data/raw
```

4. Ponemos el archivo con extension .pkl dentro de la carpeta "raw".

5. Ahora podremos correr los notebooks para su exploracion y comprobación. 


## Instrucciones para generar el ambiente

Las instrucciones detalladas se encuentrarn en en archivo `setup.md`

### Requisitos Previos

- Python 3.6 o superior
- pip
- make

### Instalación

1. Clona el repositorio:
   ```
   git clone https://github.com/isak792/watermarking
   cd watermark 
   ```

2. Crea y activa el entorno virtual, e instala las dependencias:

   Primero, se requiere una instalación de Python superior a version 3.6. Favor dirigirse a la página oficial para obtener instrucciones de instalación:
   `https://www.python.org/downloads/release/python-3120/`

   Posteriormente, se debe correr el siguiente código desde una terminal:

   ```bash
   python3 -m venv mlops_tme_venv
   ```

   Para dispositivos Windows, el código para activar el ambiente es el siguiente:
   ```bash
   mlops_tme_venv\Scripts\activate
   ```

   En cambio, este es el código para MacOS y Linux:
   ```bash
   source mlops_tme_venv/bin/activate
   ```

### Configuración del Entorno de Desarrollo y Dependencias

El siguiente comando se utiliza para instalar todas las dependencias.
Además, este proyecto utiliza pre-commit hooks para mantener la calidad del código. Para configurar el entorno de desarrollo:

```
make setup
```

Este comando instalará pre-commit y configurará los hooks necesarios.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         watermark and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── watermark   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes watermark a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

