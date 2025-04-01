import json
import optuna
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score
    

# Configuración de paths
current_dir = Path(__file__).parent
repo_root = current_dir.parent.parent
DATA_PATH = repo_root / "data" / "processed" / "diabetic_data_preprocessed.csv"
PARAMS_PATH = repo_root / "src" / "utils" / "best_parameters_foriteration.json"
MODELS_DIR = repo_root / "src" / "models" 

def load_data():
    """Carga y divide los datos"""
    df = pd.read_csv(DATA_PATH)
    #df = df.iloc[:, 1:]
    X = df.drop(columns=["target"])  
    y = df["target"]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_baseline_model(X_train, y_train):
    """Entrena modelo LightGBM con configuración base"""
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    model = lgb.LGBMClassifier(
        random_state=123,
        objective="binary",
        scale_pos_weight=pos_weight,
        n_estimators=100,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    return model

def cross_validate_model(model, X_train, y_train):
    """
    Realiza validación cruzada y retorna métricas de train/test.
    """
    kfold = KFold(n_splits=5)
    cv = cross_validate(
        model, 
        X_train, 
        y_train, 
        cv=kfold, 
        return_train_score=True, 
        scoring=['roc_auc']
    )
    
    print('\nValidación Cruzada:')
    print('test *****')
    print('valores validacion cruzada: ', cv['test_roc_auc'])
    print('valor media validacion cruzada: ', np.mean(cv['test_roc_auc']))
    print('train *****')
    print('valores validacion cruzada: ', cv['train_roc_auc'])
    print('valor media validacion cruzada: ', np.mean(cv['train_roc_auc']))
    
    return cv


def optimize_hyperparams(X_train, y_train, n_trials):
    """Optimización con Optuna"""
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    def objective(trial):

        params = {
            
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100, step=10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100, step=10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
           
        }

        model = lgb.LGBMClassifier(
            objective="binary",
            random_state=123,
            scale_pos_weight=pos_weight,
            **params
        )

        kf = KFold(n_splits=5, shuffle=True, random_state=123)
        cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring="roc_auc", n_jobs=-1)
        return np.mean(cv_results["test_score"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["scale_pos_weight"] = pos_weight  

    return best_params  

def train_final_model(X_train, y_train, best_params):
    """Entrena modelo final con mejores hiperparámetros"""
    #print('best_params: ',best_params)
    model = lgb.LGBMClassifier(
        objective="binary",
        random_state=123,
        **best_params
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo con las métricas seleccionadas:
    - Accuracy
    - Sensibilidad (recall para clase positiva)
    - Especificidad (recall para clase negativa)
    - ROC-AUC
    """

    y_pred = model.predict(X_test)
    
    print('\nMétricas de evaluación:')
    print('-' * 40)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'Sensibilidad: {recall_score(y_test, y_pred, pos_label=1):.4f}')
    print(f'Especificidad: {recall_score(y_test, y_pred, pos_label=0):.4f}')
    print(f'ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}')
    print('-' * 40)


if __name__ == "__main__":

    try:
        # Flujo completo
        X_train, X_test, y_train, y_test = load_data()
        
        # 1. Modelo baseline
        baseline_model = train_baseline_model(X_train, y_train)
        print("\n- Definicion y evaluacion del modelo base:")
        evaluate_model(baseline_model, X_test, y_test)
        print("\n OK ")

        print("\n- Evaluación por validacion cruzada del modelo base:")
        cv_baseline = cross_validate_model(baseline_model, X_train, y_train)
        print("\n OK ")
        
        # 2. Optimización
        print("\n- Optimizacion de hiper parametros:")
        best_params = optimize_hyperparams(X_train, y_train, n_trials=20)
        print("\n Mejores hiperparámetros encontrados:")
        print(best_params)
        print("\n OK ")

        # 3. Modelo final
        final_model = train_final_model(X_train, y_train, best_params)
        print("\n- Evaluación del modelo optimizado:")
        evaluate_model(final_model, X_test, y_test)
        print("\n OK ")

        # Guardar los mejores hiper parametros de los entrnemientos.
        print("\n- Guardando los mejores hiper parametros en utils/best_parameters_foriteration:")
        with open(PARAMS_PATH, "w") as f:
            json.dump(best_params, f, indent=4)  
        print("\n OK ")


        #Guardar modelo usando joblib, suele ser mas eficiente que usar directamente pickle
        print("\n- Guardando modelo optimizado en models/")
        joblib.dump(final_model, MODELS_DIR / "modelo_LGBMClassifier4.pkl") 
        print("\n OK ")

    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        raise    

