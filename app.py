import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# Configuración de página
st.set_page_config(
    page_title="Predicción RUL - NASA C-MAPSS",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Sistema de Mantenimiento Predictivo & Estimación de RUL")
st.markdown("### Monitoreo de Salud Térmica y Degradación en Motores Turbofán (Dataset FD001)")
st.write("---")

# Carga de Artefactos
@st.cache_resource
def load_artifacts():
    try:
        model_path = os.path.join("models", "best_model_fd001.pkl")
        scaler_path = os.path.join("data", "processed", "scaler_fd001.pkl")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Error cargando modelos desde disco: {e}")
        return None, None

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

model, scaler = load_artifacts()
df = load_data()

if model is not None and df is not None:
    id_col = next((col for col in ['unit_nr', 'engine_id', 'id'] if col in df.columns), df.columns[0])
    cycle_col = next((col for col in ['time_cycles', 'cycle', 'time'] if col in df.columns), df.columns[1])
    
    # Sidebar
    st.sidebar.header("⚙️ Panel de Control de Inspección")
    engine_list = sorted(df[id_col].unique())
    selected_engine = st.sidebar.selectbox("Seleccionar ID de Motor:", engine_list)
    
    engine_data = df[df[id_col] == selected_engine].sort_values(cycle_col)
    max_cycle = int(engine_data[cycle_col].max())
    selected_cycle = st.sidebar.slider("Ciclo de Vuelo Actual:", min_value=1, max_value=max_cycle, value=max_cycle)
    
    current_row = engine_data[engine_data[cycle_col] == selected_cycle].copy()
    
    # 1. Definir exactamente las 21 columnas para el Scaler
    if hasattr(scaler, "feature_names_in_"):
        scaler_cols = list(scaler.feature_names_in_)
    else:
        scaler_cols = [c for c in df.columns if c not in [id_col, cycle_col, 'RUL', 'Unnamed: 0']]
        if len(scaler_cols) != 21:
            candidates = [c for c in df.columns if 'sensor' in c.lower() or 'setting' in c.lower()]
            scaler_cols = candidates[:21]

    # Construir X_scaler (21 columnas)
    X_scaler_input = pd.DataFrame(index=[0])
    for col in scaler_cols:
        val = current_row[col].values[0] if col in current_row.columns else 0.0
        X_scaler_input[col] = float(val) if pd.notnull(val) else 0.0

    # Escalar a 21 dimensiones
    scaled_array_21 = scaler.transform(X_scaler_input.values)
    scaled_df_21 = pd.DataFrame(scaled_array_21, columns=scaler_cols)

    # 2. Extraer las 17 columnas exactas requeridas por el Modelo XGBoost
    if hasattr(model, "feature_names_in_"):
        model_cols = list(model.feature_names_in_)
    else:
        # Fallback: descartar los sensores sin varianza conocidos en C-MAPSS FD001 (1, 5, 10, 16, 18, 19, setting_3)
        cols_to_drop = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19', 'setting_3']
        model_cols = [c for c in scaler_cols if c not in cols_to_drop]
        if len(model_cols) != 17:
            model_cols = scaler_cols[:17]

    # Filtrar scaled_df a las 17 columnas del modelo
    X_model_input = pd.DataFrame()
    for col in model_cols:
        if col in scaled_df_21.columns:
            X_model_input[col] = scaled_df_21[col].values
        else:
            X_model_input[col] = 0.0

    # Inferencia
    try:
        rul_pred = float(model.predict(X_model_input)[0])
    except Exception:
        rul_pred = float(model.predict(X_model_input.values)[0])

    # Métricas Principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Motor Bajo Análisis", f"Unidad #{selected_engine}")
    col2.metric("Ciclo Evaluado", f"{selected_cycle} / {max_cycle}")
    col3.metric("RUL Estimado (XGBoost)", f"{rul_pred:.1f} ciclos")
    
    st.write("---")
    
    # Diagnóstico de Mantenimiento Predictivo
    st.subheader("🔴 / 🟡 / 🟢 Diagnóstico de Mantenimiento Predictivo")
    if rul_pred > 45:
        st.success(f"🟢 **ESTADO NOMINAL:** El motor #{selected_engine} posee un margen seguro de {rul_pred:.1f} ciclos remanentes. Operación continua aprobada.")
    elif 20 < rul_pred <= 45:
        st.warning(f"🟡 **ADVERTENCIA:** Vida útil remanente crítica ({rul_pred:.1f} ciclos). Se recomienda programar inspección boroscópica en el HPC.")
    else:
        st.error(f"🔴 **ALERTA MÁXIMA:** Retiro inmediato recomendado. RUL estimado de {rul_pred:.1f} ciclos cercano al umbral de falla funcional.")
        
    st.write("---")
    
    # Gráfico Dinámico de Telemetría
    st.subheader("📊 Histórico de Telemetría y Monitoreo de Sensores Críticos")
    sensor_options = [c for c in df.columns if 'sensor' in c.lower()]
    if sensor_options:
        sensor_to_plot = st.selectbox("Seleccionar Sensor para Inspeccionar Visualmente:", sensor_options)
        fig = px.line(engine_data, x=cycle_col, y=sensor_to_plot, title=f"Evolución de {sensor_to_plot} en Motor #{selected_engine}")
        fig.add_vline(x=selected_cycle, line_dash="dash", line_color="red", annotation_text="Ciclo Seleccionado")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Verificando la disponibilidad de los artefactos en data/processed/ y models/...")
