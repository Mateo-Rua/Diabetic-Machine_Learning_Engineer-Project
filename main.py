from pathlib import Path
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from src.preprocessing.clean_data import main_clean
from src.train.train_models import train_final_model, evaluate_model
from src.predict.predictor import predict, save_model_metadata
from src.utils.config import DATA_PATHS  

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    try:
        # 1. Cargamos y limpieza
        logger.info("Cargando datos crudos...")
        raw_data_path = Path(DATA_PATHS['raw'])
        raw_data = pd.read_csv(raw_data_path)
        
        logger.info("Ejecutando limpieza de datos...")
        cleaned_data = main_clean(raw_data)
        logger.info(f"Datos limpiados. Forma: {cleaned_data.shape}")

        # 2. Preparamos de datos
        logger.info("Dividiendo datos en entrenamiento y prueba...")
        X = cleaned_data.drop("readmitted", axis=1)
        y = cleaned_data["readmitted"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            stratify=y,  # Mantener distribución de clases
            random_state=123
        )

        # 3. Entrenamiento del modelo
        logger.info("Entrenando modelo final...")
        model = train_final_model(X_train, y_train)
        
        # 4. Evaluación
        logger.info("Evaluando modelo...")
        evaluate_model(model, X_test, y_test)
        
        # 5..  OJOOOO ESTE ES SOLO UN EJEMPLO DE PREDICCION
        logger.info("Generando predicciones de ejemplo...")
        sample_data = X_test.iloc[:5].copy()
        predictions = predict(sample_data)
        logger.info(f"Predicciones de ejemplo:\n{pd.Series(predictions, index=sample_data.index)}")
        
        # 6. Guardarmos recursos
        logger.info("Guardando artefactos del modelo...")
        model_path = Path(DATA_PATHS['models']) / "trained_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardarmos modelo con metadatos
        save_model_metadata(
            model=model,
            model_path=model_path,
            feature_names=list(X_train.columns),
            model_version="1.0.0"
        )
        
        logger.info("Pipeline completado exitosamente!")

    except Exception as e:
        logger.error(f"Error en el pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_pipeline()