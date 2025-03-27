from lightgbm import LGBMClassifier
from src.utils.config import BEST_PARAMS  # Hiperparámetros desde config.py

def create_model(pos_weight: float):
    """
    Crea y retorna el modelo optimizado con los mejores hiperparámetros.
    Args:
        pos_weight: Peso para la clase positiva (calculado en train_models.py)
    """
    return LGBMClassifier(
        objective="binary",
        random_state=123,
        scale_pos_weight=pos_weight,
        **BEST_PARAMS
    )