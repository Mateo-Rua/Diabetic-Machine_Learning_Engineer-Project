import joblib
import pandas as pd
from pathlib import Path

def load_model():
    """Carga el modelo entrenado"""
    model_path = Path(__file__).parent.parent / "models" / "modelo_LGBMClassifier.pkl"
    return joblib.load(model_path)

def predict(input_data: pd.DataFrame):
    """
    Realiza predicciones con nuevos datos.
    Args:
        input_data: DataFrame con las mismas features que usó el modelo
        en face de entenamiento.
    Returns:
        Array con las predicciones (0 o 1).
    """
    model = load_model()
    return model.predict(input_data)



# Ejemplo de como usar la prediccion del modelo
"""
if __name__ == "__main__":
    # Ejemplo con datos que simulan tu esquema real
    test_data = pd.DataFrame({
        "age": [5],  # Ejemplo: valor codificado
        "A1Cresult": [7.0],
        # ... (todas las features usadas en entrenamiento)
    })
    print("Predicción:", predict(test_data))"

"""