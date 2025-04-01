import argparse
import sys
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuración de rutas
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from src.preprocessing.clean_data import main_clean
from src.train.traint_models import cross_validate_model,optimize_hyperparams, train_final_model,evaluate_model

# Rutas clave
models_dir = current_dir / "src" / "models"
model_path = models_dir / "modelo_LGBMClassifier_final4.pkl"
model_path_predict = models_dir / "modelo_LGBMClassifier_final.pkl"
DATA_PATH = current_dir / "data" / "processed" / "diabetic_data_preprocessed.csv"

def main(args):
    try:
        if args.modo == "full":
            # --------------------------------------------
            # MODO FULL: Entrenamiento + Preprocesamiento
            # --------------------------------------------
            print("\n Preprocesando datos...")
            raw_df = pd.read_csv(
                current_dir / "data" / "raw" / "diabetic_data.csv"
            )
            cleaned_df = main_clean(raw_df)
            cleaned_df = pd.read_csv(current_dir / "data" / "processed" / "diabetic_data_preprocessed.csv")
            #cleaned_df.to_csv(current_dir / "data" / "processed" / "diabetic_data_preprocessed.csv", index=False)
            
            print("\n Definimos Features  X , y...")
            X = cleaned_df.drop(columns=["target"])
            y = cleaned_df["target"]
            
            print("\n Train_test_split...")
            # Split de datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.3, 
                random_state=42,
                stratify=y
            )
            

            print("\n Optimizando hiperparámetros...")
            # Optimizar hiperparámetros y entrenar
            best_params = optimize_hyperparams(X_train, y_train, n_trials=5)

            print("\n Entrenando modelo...")
            final_model = train_final_model(X_train, y_train, best_params)

            print("\n Validacion cruzada...")
            cross_validate_model(final_model, X_train, y_train)
            
            print("\n Evaluacion del modelo...")
            evaluate_model(final_model, X_test, y_test)
            # Guardar modelo
            joblib.dump(final_model, model_path)
            print(f" Modelo guardado en: {model_path}")

        elif args.modo == "predict":

            # MODO PREDICT: Solo predicciones
         
            print("\n Realizando predicciones...")
            # Cargar modelo y datos
            modelo = joblib.load(model_path_predict) 
            cleaned_data = pd.read_csv(current_dir / "data" / "test" / "test_samples.csv")
            #X_tesst = cleaned_data.drop(columns=["target"]).iloc[:, :50]
            
            # Predecir y guardar
            predictions = modelo.predict(cleaned_data)
            #df_compare = DATA_PATH['target']
            df_target = pd.read_csv(DATA_PATH)
            target_series = df_target['target'].head(50).squeeze()

            predictions_df = pd.DataFrame({
                'Predicción': predictions,
                'Target Real': target_series
            })
            
            # Guardar en CSV
            predictions_df.to_csv(args.output_file, index=False)
            
            # imprimer comparacion entre predicciones y target real
            print("\n Predicciones realizadas:")
            print(predictions_df.head(20))  # Muestra las primeras 10 predicciones
            pd.DataFrame(predictions, columns=["prediccion"]).to_csv(args.output_file, index=False)
            print(f" Predicciones guardadas en: {args.output_file}")

    except Exception as e:
        print(f"\n Error: {str(e)}")
        raise

if __name__ == "__main__":
    # Aqui arreglo los argumentos para ejecutar las funciones
    parser = argparse.ArgumentParser(description="Pipeline de ML: Entrenar o predecir")
    parser.add_argument(
        "--modo", 
        type=str, 
        choices=["full", "predict"], 
        required=True,  
        help="Modo de ejecución: 'full' (entrenar) o 'predict' (predecir)"
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="data/test/test_samples.csv",
        help="Ruta a archivo CSV para predicciones"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="data/test/predictions.csv",
        help="Ruta para guardar resultados"
    )
    
    args = parser.parse_args()
    main(args)





