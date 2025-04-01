import sys
import json
import joblib
from pathlib import Path
from lightgbm import LGBMClassifier

current_dir = Path(__file__).parent  # src/models/
repo_root = current_dir.parent.parent  # Raíz del proyecto

# Añadimos el repo root al sys.path (IMPORTANTE para imports)
sys.path.insert(0, str(repo_root))

# importamos las funciones dek archivo traint_models.py
from src.train.traint_models import load_data, evaluate_model 
from src.utils.config import BEST_PARAMS

MODELS_DIR = current_dir 

def create_model():
    return LGBMClassifier(
        objective="binary",
        random_state=123,
        **BEST_PARAMS
    )

def train_and_save_model():
    """ Se entrena y se guarda el modelo
     con los hiperparametros personalizados """ 
    try:
        X_train, X_test, y_train, y_test = load_data()
        model = create_model()
        print("Entrenando modelo...")
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        
        output_path = MODELS_DIR / "modelo_LGBMClassifier_final2.pkl"
        joblib.dump(model, output_path)
        print(f"Modelo guardado en: {output_path}") 
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model()