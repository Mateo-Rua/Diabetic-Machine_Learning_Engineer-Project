from pathlib import Path
import pandas as pd
import sys

# Configuración de paths
current_dir = Path(__file__).parent  # src/preprocessing/
src_dir = current_dir.parent  # src/
sys.path.append(str(src_dir))  
from utils.constants import list_order, list_numeric_int_Deabetic,medicamentos,ML_training_features


repo_root = current_dir.parent.parent  
raw_data_path = repo_root / "data" / "raw" / "diabetic_data.csv"

def ordenar_df(df,lista):
    """  
    Ordenar la estructura del nuevo dataset,
    asociando la informacion  que se describe de los datos.
    """
    df = df.copy()
    df = df[lista]
    return df


def delete_abcde(df):
    """"
    Eliminar las columnas que contienen datos abcde.
    """
    df = df.copy()
    df = df[~(df.apply(lambda row: (row == "abcde").all(), axis=1))]
    return df


def convert_object_to_int(df, list_numeric_int):
    """
    Transformar los datos de tipo object a int64.
    """
    cols_existentes = [col for col in list_numeric_int if col in df.columns]
    df[cols_existentes] = df[cols_existentes].apply(pd.to_numeric, errors='coerce').astype("Int64")
    return df


def map_target(df):
    """
    Mapeo del targe.
    """
    df['target'] = df['readmitted'].map({
        '<30': 1,    # Caso positivo (readmisión temprana)
        '>30': 0,    # Caso negativo
        'NO': 0      # Caso negativo
    })
    return df


def replace_race(df):
    """
    Comvertir ? a Other de la columna race.
    """
    df['race'] = df['race'].replace('?', 'Other')
    return df


def exclude_unknown_and_NaN(df):
    """
    Excluir registros con valores nulos y 'Desconocido/Inválido
    """
    exclude_indexes = df[df['gender'] == 'Unknown/Invalid'].index.tolist()
    exclude_indexes = exclude_indexes + df[df['diag_1'].isna()].index.tolist()
    exclude_indexes = exclude_indexes + df[df['diag_2'].isna()].index.tolist()
    exclude_indexes = exclude_indexes + df[df['diag_3'].isna()].index.tolist()
    required_indexes = [index for index in df.index.tolist() if index not in list(set(exclude_indexes))]
    df = df.loc[required_indexes]
    return df


def drop_columns(df):
    """
    Elimina por completo las columnas que contienen en su mayoria datos NaN o con un unico valor
    que no aporta mucha informacion al modelo.
    """
    df = df.drop(['weight','payer_code','medical_specialty','examide', 'citoglipton', 'glimepiride-pioglitazone'], axis = 1)
    return df


def transform_and_categorize(df):
  """
  Vamos a categorizar algunas de las columnas de tipo objeto
  y vamos a tnansforma algunas categorias a valores números enteros.
  """
  df['race'] = df['race'].astype('category').cat.codes

  df['gender'] = df['gender'].replace('Male', 1)
  df['gender'] = df['gender'].replace('Female', 0)

  df['age'] = df['age'].replace('[0-10)', 0)
  df['age'] = df['age'].replace('[10-20)', 1)
  df['age'] = df['age'].replace('[20-30)', 2)
  df['age'] = df['age'].replace('[30-40)', 3)
  df['age'] = df['age'].replace('[40-50)', 4)
  df['age'] = df['age'].replace('[50-60)', 5)
  df['age'] = df['age'].replace('[60-70)', 6)
  df['age'] = df['age'].replace('[70-80)', 7)
  df['age'] = df['age'].replace('[80-90)', 8)
  df['age'] = df['age'].replace('[90-100)', 9)

  df['diag_1'] = df['diag_1'].astype('category').cat.codes
  df['diag_2'] = df['diag_2'].astype('category').cat.codes
  df['diag_3'] = df['diag_3'].astype('category').cat.codes
  return df


def process_A1Cresult(df):
  """
  Procesar y Tranformar los datos de la columna A1Cresult
  segun el analisis y la experiemntacion con los datos.
  """
  df['A1Cresult'] = df['A1Cresult'].replace('>7', 2)
  df['A1Cresult'] = df['A1Cresult'].replace('>8', 2)
  df['A1Cresult'] = df['A1Cresult'].replace('Norm', 1)
  df['A1Cresult'] = df['A1Cresult'].fillna(0)
  return df

 
def process_max_glu_serum(df):
  """
  Procesar y Tranformar los datos de la columna max_glu_serum
  segun el analisis y la experiemntacion con los datos.
  """
  df['max_glu_serum'] = df['max_glu_serum'].replace('>200', 2)
  df['max_glu_serum'] = df['max_glu_serum'].replace('>300', 2)
  df['max_glu_serum'] = df['max_glu_serum'].replace('Norm', 1)
  df['max_glu_serum'] = df['max_glu_serum'].fillna(0)
  return df


def process_medicamentos(df,medication_list):
  """
  Procesar o binarizar los datos de las columnas que llevan el nombre de algun medicamento.
  """
  for col in medication_list:
      df[col] = df[col].replace('No', 0)
      df[col] = df[col].replace('Steady', 1)
      df[col] = df[col].replace('Up', 1)
      df[col] = df[col].replace('Down', 1)
  return df


def process_change(df):
    """
    Binarizar los datos de la columna change 
    segun el analisis de los datos
    para que el modelo pueda generalizar mejor la informacion. 
    """
    df['change'] = df['change'].replace('Ch', 1)
    df['change'] = df['change'].replace('No', 0)
    return df


def process_diabetesMed(df):
    """
    Binarizar los datos de la columna diabetesMed
    segun el analisis de los datos
    para que el modelo pueda generalizar mejor la informacion. 
    """
    df['diabetesMed'] = df['diabetesMed'].replace('Yes', 1)
    df['diabetesMed'] = df['diabetesMed'].replace('No', 0)
    return df




def main_clean(df):
    """
    Limpieza y preprocesamiento del dataset diabético.
    
    Args:
        df (pd.DataFrame): Datos crudos con columnas originales.
        
    Returns:
        pd.DataFrame: Datos procesados listos para modelado.
    """
    df = df.copy()
    df = ordenar_df(df,list_order)
    df = delete_abcde(df)
    df = convert_object_to_int(df, list_numeric_int_Deabetic)
    df = map_target(df)
    df = replace_race(df)
    df = exclude_unknown_and_NaN(df)
    df = drop_columns(df)
    df = transform_and_categorize(df)
    df = process_A1Cresult(df)
    df =process_max_glu_serum(df)
    df = process_medicamentos(df,medicamentos)
    df = process_change(df)
    df = process_diabetesMed(df)
    df = ordenar_df(df,ML_training_features)
    return df



if __name__ == "__main__":

    
    try:
        print("\n- Preprocesando data:")
        #raw_df = pd.read_csv(raw_data_path)
        raw_df = pd.read_csv(raw_data_path, dtype='object')  
        cleaned_df = main_clean(raw_df)
        print("\n OK ")

        print("\n- Guardando data preprocesada:")
        # Guarda el DataFrame procesado
        processed_path = repo_root / "data" / "processed" / "diabetic_data_preprocessed2.csv"
        cleaned_df.to_csv(processed_path, index=False)
        print("\n OK ")

    except Exception as e:
        raise Exception(f"Error: {str(e)}")  

