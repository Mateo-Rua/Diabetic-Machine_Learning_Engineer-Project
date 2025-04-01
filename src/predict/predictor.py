import joblib
import pandas as pd
from pathlib import Path

# Configuración de rutas 
model_path = Path(__file__).parent.parent / "models" / "modelo_LGBMClassifier_final.pkl"
test_path = Path(__file__).parent.parent.parent / "data" / "test" / "test_samples.csv"

def load_model():
    #Carga el modelo entrenado
    if not model_path.exists():
        raise FileNotFoundError(f" Modelo no encontrado en: {model_path}")
    return joblib.load(model_path)

def predict(input_data: pd.DataFrame):
    
    #cargar modelo
    model = load_model()
    
    #predicciones
    predictions = model.predict(input_data)
    
    # Crear DataFrame con resultados
    result_df = input_data.copy()
    result_df["prediccion"] = predictions
    return result_df

if __name__ == "__main__":
    try:
        # 1. Cargamod datos de prueba
        test_data = pd.read_csv(test_path)
        
        # 2. Hacemos predicciones
        resultado = predict(test_data)
        
        # 3.Mostramos resultados
        print("\n Predicciones exitosas (primeras 10 filas):")
        print(resultado[["prediccion"]].head(20))
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        
        
