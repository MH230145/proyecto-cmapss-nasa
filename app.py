import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# Configuración inicial de la página
st.set_page_config(
    page_title="Predicción RUL - NASA C-MAPSS",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Sistema de Mantenimiento Predictivo & Estimación de RUL")
st.markdown("### Monitoreo de Salud Térmica y Degradación en Motores Turbofán (Dataset FD001)")
st.write("---")

# Carga de Artefactos de Modelo
@st.cache_resource
def load_artifacts():
    try:
        model_path = os.path.join("models", "best_model_fd001.pkl")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Error cargando el modelo desde disco: {e}")
        return None

@st.cache_data
def load_data():
    possible_paths = [
        os.path.join("data", "processed", "Train_processed_df.csv"),
        os.path.join("data", "processed", "train_processed_df.csv")
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                return df
            except Exception as e:
                st.error(f"⚠️ Error al leer {path}: {e}")
                return None
    st.error("⚠️ No se encontró el archivo CSV procesado.")
    return None

model = load_artifacts()
df = load_data()

if model is not None and df is not None:
    # Identificación de columnas principales
    id_col = next((col for col in ['engine_id', 'unit_nr', 'id'] if col in df.columns), df.columns[0])
    cycle_col = next((col for col in ['cycle', 'time_cycles', 'time'] if col in df.columns), df.columns[1])
    
    # Sidebar: Selección de Unidad y Ciclo
    st.sidebar.header("⚙️ Panel de Control de Inspección")
    engine_list = sorted(df[id_col].unique())
    selected_engine = st.sidebar.selectbox("Seleccionar ID de Motor:", engine_list)
    
    engine_data = df[df[id_col] == selected_engine].sort_values(cycle_col)
    max_cycle = int(engine_data[cycle_col].max())
    selected_cycle = st.sidebar.slider("Ciclo de Vuelo Actual:", min_value=1, max_value=max_cycle, value=max_cycle)
    
    # Fila de datos en el ciclo seleccionado
    current_row = engine_data[engine_data[cycle_col] == selected_cycle].copy()
    
    # 1. Extraer nombres exactos de características requeridas por XGBoost
    if hasattr(model, "feature_names_in_"):
        model_cols = list(model.feature_names_in_)
    else:
        # Si no hay metadatos, ignorar columnas meta/id/target
        ignore_cols = [id_col, cycle_col, 'RUL', 'RUL_clipped', 'rot_RUL_clipped', 'Unnamed: 0']
        model_cols = [c for c in df.columns if c not in ignore_cols]
        if len(model_cols) > 17:
            model_cols = model_cols[:17]

    # 2. Construir DataFrame de entrada directamente con los datos procesados del CSV (Sin re-escalar)
    X_input = pd.DataFrame(index=[0])
    for col in model_cols:
        val = current_row[col].values[0] if col in current_row.columns else 0.0
        X_input[col] = float(val) if pd.notnull(val) else 0.0

    # 3. Inferencia directa del RUL
    try:
        rul_pred = float(model.predict(X_input)[0])
    except Exception:
        rul_pred = float(model.predict(X_input.values)[0])

    # Si la variable objetivo RUL en tu modelo se entrenó normalizada o clipping 0-125
    # Garantizamos que la lectura no devuelva valores negativos
    rul_pred = max(0.0, rul_pred)

    # Métricas Principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Motor Bajo Análisis", f"Unidad #{selected_engine}")
    col2.metric("Ciclo Evaluado", f"{selected_cycle} / {max_cycle}")
    col3.metric("RUL Estimado (XGBoost)", f"{rul_pred:.1f} ciclos")
    
    st.write("---")
    
    # Diagnóstico de Mantenimiento Predictivo
    st.subheader("Diagnóstico de Mantenimiento Predictivo")
    if rul_pred > 45:
        st.success(f"**ESTADO NOMINAL:** El motor #{selected_engine} posee un margen seguro de {rul_pred:.1f} ciclos remanentes. Operación continua aprobada.")
    elif 20 < rul_pred <= 45:
        st.warning(f"**ADVERTENCIA:** Vida útil remanente crítica ({rul_pred:.1f} ciclos). Se recomienda programar inspección boroscópica en el HPC.")
    else:
        st.error(f"**ALERTA MÁXIMA:** Retiro inmediato recomendado. RUL estimado de {rul_pred:.1f} ciclos cercano al umbral de falla funcional.")
        
    st.write("---")
    
    # Gráfico Dinámico de Telemetría
    st.subheader("Histórico de Telemetría y Monitoreo de Sensores Críticos")
    sensor_options = [c for c in df.columns if 'sensor' in c.lower()]
    if sensor_options:
        sensor_to_plot = st.selectbox("Seleccionar Sensor para Inspeccionar Visualmente:", sensor_options)
        fig = px.line(engine_data, x=cycle_col, y=sensor_to_plot, title=f"Evolución de {sensor_to_plot} en Motor #{selected_engine}")
        fig.add_vline(x=selected_cycle, line_dash="dash", line_color="red", annotation_text="Ciclo Seleccionado")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Verificando la disponibilidad de los artefactos en data/processed/ y models/...")
