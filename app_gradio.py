import gradio as gr
import pandas as pd
import joblib
from pathlib import Path

# Configurar rutas
current_dir = Path(__file__).parent
model_path = current_dir / "src" / "models" / "modelo_LGBMClassifier_final.pkl"
data_sample_path = current_dir / "data" / "processed" / "diabetic_data_preprocessed.csv"

# Cargar modelo y datos de ejemplo
model = joblib.load(model_path)
sample_data = pd.read_csv(data_sample_path)

def predict_csv(csv_file):
    # Leer y procesar el CSV
    df = pd.read_csv(csv_file)
    
    # Verificar si existe el target real
    has_real_target = "target" in df.columns
    
    # Hacer predicciones
    predictions = model.predict(df.drop(columns=["target"] if has_real_target else df))
    
    # Crear DataFrame de resultados
    results = pd.DataFrame({
        "ID_Paciente": df.index + 1,
        "Predicción": ["1 Alto riesgo" if p == 1 else "0 Bajo riesgo" for p in predictions]
    })
    
    # Agregar target real
    if has_real_target:
          results["Target_Real"] = df["target"].apply(lambda x: "1 Alto riesgo" if x == 1 else "0 Bajo riesgo")
    
    return results

def predict_single(*inputs):
    # Crear DataFrame desde los inputs
    input_df = pd.DataFrame([inputs], columns=sample_data.columns.drop("target"))
    
    # Hacer predicción
    prediction = model.predict(input_df)[0]
    return "1 - Alto riesgo de readmisión" if prediction == 1 else "0 - Bajo riesgo de readmisión"

# Interfaz para archivos CSV
csv_interface = gr.Interface(
    fn=predict_csv,
    inputs=gr.File(file_types=[".csv"], label="Subir CSV"),
    outputs=gr.Dataframe(
        headers=["ID_Paciente", "Predicción", "Target_Real"] if "target" in sample_data.columns else ["ID_Paciente", "Predicción"],
        datatype=["str", "str", "str"] if "target" in sample_data.columns else ["str", "str"]
    ),
    title="Predicciones por archivo CSV",
    #examples=[data_sample_path] if Path(data_sample_path).exists() else None
    examples=[str(data_sample_path)] if data_sample_path.exists() else None
    
)

# Interfaz para predicción individual
single_interface = gr.Interface(
    fn=predict_single,
    inputs=[gr.Number(label=col) for col in sample_data.columns if col != "target"],
    outputs=gr.Label(label="Resultado"),
    title="Predicción individual",
    examples=[
        sample_data.drop(columns=["target"]).iloc[0].tolist(),
        sample_data.drop(columns=["target"]).iloc[46].tolist()
    ]
)

# Aplicación completa con pestañas
app = gr.Blocks(title="Sistema de Predicción de Readmisión Hospitalaria")
with app:
    gr.Markdown("# Predice cuales son los posibles pacientes con riesgo de readmisión en 30 días")
    gr.Markdown("Predictor de Readmisión Hospitalaria")
    
    with gr.Tabs():
        with gr.TabItem(" Predicción por archivo CSV"):
            csv_interface.render()
        
        with gr.TabItem(" Predicción individual"):
            single_interface.render()

if __name__ == "__main__":
    app.launch(
        server_port=7860,
        share=True,
        favicon_path="https://raw.githubusercontent.com/gradio-app/gradio/main/guides/assets/favicon.ico"
    )


