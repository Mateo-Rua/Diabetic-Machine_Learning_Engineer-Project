# Readmisión en Pacientes Diabéticos.

En el sigueinte proyecto se analizara un  dataset que representa los datos de varios años de atención clínica a personas diabéticas en varios hospitales de EEUU. El dataset tiene múltiples
características que representan los resultados del paciente y del hospital o centro de atención.

## El objetivo de este proyecto.

Desarrollar un modelo de predicción que determine si unpaciente será readmitido en el hospital en menos de 30 días.

## Contexto del proyecto.

Este proyecto forma parte del proceso de selección para la vacante de ML Engineer. En él, se aplicarán diversas técnicas y métodos, incluyendo:

- Procesamiento y análisis de datos
- Feature Engineering
- Modelo predictivo
- Métricas y validacion del modelo
- Versionado de código
- Skills en Python

## Estructura del proyaecto. 
```
|   main.py
|   README.md
|   requirements.txt
|
+---data
|   +---processed
|   |       diabetic_data_preprocessed.csv
|   |
|   \---raw
|           diabetic_data.csv
|
+---notebooks
|       notebook_diabetic_data.ipynb
|
\---src
    |   __init__.py
    |
    +---models
    |       best_model.py
    |       modelo_catboost.pkl
    |       modelo_LGBMClassifier.pkl
    |
    +---predict
    |       predictor.py
    |
    +---preprocessing
    |       clean_data.py
    |
    +---train
    |       traint_models.py
    |
    \---utils
            best_parameters_foriteration.json
            config.py
            constants.py

```
## Features
