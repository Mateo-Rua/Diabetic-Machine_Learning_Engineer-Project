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


### data/

### raw: 
Datos originales descargados o recibidos (no modificados).

### processed:
Datos procesados listos para entrenar modelos (limpieza, transformaciones).

### notebooks/

Análisis exploratorios (EDA), pruebas de conceptos, o visualizaciones intermedias.

### src/

### models: 
Modelos serializados (.pkl) y lógica para gestionarlos.

### predict: 
Funcionalidad para hacer predicciones en producción.

### preprocessing: 
Transformación de datos crudos a datos listos para modelos.

### train:
Scripts de entrenamiento de modelos (incluyendo validación).

### utils:
Configuraciones, constantes y parámetros reutilizables.

### main.py

- Automatiza todo el flujo de trabajo del proyecto:

- Carga datos crudos desde data/raw/

- Ejecuta el preprocesamiento (src/preprocessing/clean_data.py)

- Entrena modelos con los parámetros óptimos (src/train/train_models.py)

- Evalúa métricas de performance

- Guarda artefactos finales en src/models/



### app_gradio.py

- Interfaz de usuario para demostrar el modelo en producción:

- Carga el modelo entrenado (src/models/modelo_catboost.pkl)

- Muestra predicciones de riesgo de diabetes (probabilidad y clasificación binaria)

