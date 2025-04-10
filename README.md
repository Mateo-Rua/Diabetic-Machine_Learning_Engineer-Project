# Readmisión en Pacientes Diabéticos.

En el sigueinte proyecto se analizara un  dataset que representa los datos de varios años de atención clínica a personas diabéticas en varios hospitales de EEUU. El dataset tiene múltiples
características que representan los resultados del paciente y del hospital o centro de atención.

El conjunto de datos representa diez años (1999-2008) de atención clínica en 130 hospitales de Estados Unidos y redes de entrega integradas. Cada fila se refiere a los registros hospitalarios de pacientes diagnosticados con diabetes, que se sometieron a laboratorio, medicamentos y permanecieron hasta 14 días. El objetivo es determinar la readmisión temprana del paciente dentro de los 30 días posteriores al alta. El problema es importante por las siguientes razones. A pesar de la evidencia de alta calidad que muestra mejores resultados clínicos para los pacientes diabéticos que reciben diversas intervenciones preventivas y terapéuticas, muchos pacientes no los reciben. Esto puede atribuirse parcialmente al manejo arbitrario de la diabetes en entornos hospitalarios, que no atienden el control glucémico.La falta de atención adecuada para la diabetes no solo aumenta los costos de administración de los hospitales (ya que los pacientes son readmitidos) sino que también afecta la morbilidad y la mortalidad de los pacientes, que pueden enfrentar complicaciones asociadas con la diabetes.

## El objetivo de este proyecto.

Desarrollar un modelo de predicción que determine si unpaciente será readmitido en el hospital en menos de 30 días.

## Contexto del proyecto.

En eeste proyecto de ML Engineer se va a realizar primero en EDA. posteriormente la experiemtacion con distintos modelos, luego el codigo sera modularizado en el repositorio y por ultimo sera desplegado en una aplicacion. En él, se aplicarán diversas técnicas y métodos, incluyendo:

- Procesamiento, análisis y limpieza de los datos
- Feature Engineering
- Modelado predictivo
- Métricas y validacion del modelo
- Optimizacion de hiperparametros
- Versionado de código
- Despliegue en la aplicacion
  

## Estructura del proyaecto. 
```
.
├── main.py                 # Punto de entrada principal para ejecutar pipelines
├── app_gradio.py           # Interfaz de usuario con Gradio para probar predicciones
├── README.md               # Documentación del proyecto
├── requirements.txt        # Dependencias del proyecto
│
├── data/                   # Gestión de datos
│   ├── raw/                # Datos originales (inmutables)
│   │       diabetic_data.csv
│   ├── processed/          # Datos procesados para entrenamiento
│   │       diabetic_data_preprocessed.csv
│   └── test/              # Datos de prueba finales
│           test.csv
│
├── notebooks/              # Experimentación y análisis
│       notebook_diabetic_data.ipynb
│
└── src/                    # Código fuente modular
    ├── __init__.py         # Identifica el directorio como paquete Python
    │
    ├── models/             # Modelos y evaluadores
    │       best_model.py   # Evalúa métricas del modelo óptimo
    │       modelo_catboost.pkl    # Modelo CatBoost serializado
    │       modelo_LGBMClassifier.pkl  # Modelo LightGBM serializado
    │
    ├── predict/            # Módulo de predicción
    │       predictor.py    # Carga modelos y genera predicciones
    │
    ├── preprocessing/      # Transformación de datos
    │       clean_data.py   # Pipeline de limpieza y preprocesamiento
    │
    ├── train/              # Entrenamiento de modelos
    │       traint_models.py  # Script de entrenamiento con ajuste de hiperparámetros
    │
    └── utils/              # Herramientas auxiliares
            config.py       # Configuración de rutas y parámetros
            constants.py    # Constantes del proyecto
            best_parameters_foriteration.json  # Hiperparámetros optimizados

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

- Interfaz de usuario para demostrar el modelo en producción

- Carga el modelo entrenado (src/models/modelo_catboost.pkl)

- Muestra predicciones de riesgo de diabetes (probabilidad y clasificación binaria)

## Guía de inicio

### 1. Clonar el repositorio:
```bash
# git clone: Descarga una copia del código del repositorio de GitHub a tu computadora.
git clone https://github.com/Mateo-Rua/Diabetic-Machine_Learning_Engineer-Project.git

# cd ecommerce-scraper: Entra en la carpeta del proyecto que acabas de clonar.
cd Diabetic-Machine_Learning_Engineer-Project
```

### 2. Instalar dependencias:
```bash
# Opción recomendada (modo desarrollo):
# Instala el paquete en "modo editable" (los cambios que hagas en el código se reflejarán sin reinstalar).
pip install -e 

# Opción básica:
# Instala solo las librerías listadas en este archivo (sin modo editable).
pip install -r requirements.txt
```

### 3. Ejecutar pruebas:
```bash
# Ejecuta el entrenmainto, validacion cruzada y la optimizacion de hiperparametros. 
main.py --modo full

# Ejecutar predicciones utilizando el modelo de mejor rendimiento con los datos de test.
main.py --modo predict
```

### 4. Ejecutar aplicacion:
```bash
# Probar el modelo implementado en una interfaz interactiva para validar su funcionamiento en un entorno de despliegue simulado. 
app_gradio.py
```
Qué hace este script:

- Carga el modelo preentrenado (requiere que exista en la ruta configurada).
- Inicia un servidor local con una interfaz gráfica (UI) construida con Gradio.
- Genera una URL temporal (ej. http://localhost:7860) para interactuar con el modelo desde el navegador.

### 5. Performance del modelo:
```bash
# Verificar el desempeño del modelo seleccionado mediante métricas de evaluación predefinidas.
best_model.py
```
