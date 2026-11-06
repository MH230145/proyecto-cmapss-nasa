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

# Carga de Artefactos con Manejo de Rutas Relativas
@st.cache_resource
def load_artifacts():
    try:
        model_path = os.path.join("models", "best_model_fd001.pkl")
        scaler_path = os.path.join("data", "processed", "scaler_fd001.pkl")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Error cargando modelos o escaladores desde disco: {e}")
        return None, None

@st.cache_data
def load_data():
    try:
        data_path = os.path.join("data", "processed", "train_processed_df.csv")
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"⚠️ Error cargando datos procesados desde data/processed: {e}")
        return None

# Cargar componentes
model, scaler = load_artifacts()
df = load_data()

if model is not None and df is not None:
    # Identificación dinámica de columnas de identificación y tiempo
    id_col = 'unit_nr' if 'unit_nr' in df.columns else ('engine_id' if 'engine_id' in df.columns else df.columns[0])
    cycle_col = "time_cycles" if "time_cycles" in df.columns else ("cycle" if "cycle" in df.columns else df.columns[1])
    
    # Sidebar: Selección de Unidad y Ciclo
    st.sidebar.header("⚙️ Panel de Control de Inspección")
    engine_list = sorted(df[id_col].unique())
    selected_engine = st.sidebar.selectbox("Seleccionar ID de Motor:", engine_list)
    
    engine_data = df[df[id_col] == selected_engine].sort_values(cycle_col)
    max_cycle = int(engine_data[cycle_col].max())
    selected_cycle = st.sidebar.slider("Ciclo de Vuelo Actual:", min_value=1, max_value=max_cycle, value=max_cycle)
    
    # Fila de datos en el ciclo seleccionado
    current_row = engine_data[engine_data[cycle_col] == selected_cycle]
    
    # Extracción exacta de las características usadas por el escalador durante el fit
    if hasattr(scaler, "feature_names_in_"):
        feature_cols = list(scaler.feature_names_in_)
    else:
        # Respaldos de columnas activas (14 sensores + 2 settings)
        feature_cols = ['setting_1', 'setting_2', 'sensor_2', 'sensor_3', 'sensor_4', 
                        'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 
                        'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
    
    # Garantizar presencia de todas las columnas esperadas
    X_input = current_row[feature_cols]
    
    # Predicción
    X_scaled = scaler.transform(X_input)
    rul_pred = float(model.predict(X_scaled)[0])
    
    # Métricas Principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Motor Bajo Análisis", f"Unidad #{selected_engine}")
    col2.metric("Ciclo Evaluado", f"{selected_cycle} / {max_cycle}")
    col3.metric("RUL Estimado (XGBoost)", f"{rul_pred:.1f} ciclos")
    
    st.write("---")
    
    # Diagnóstico y Alertas Tácticas (PdM)
    st.subheader("🔴 / 🟡 / 🟢 Diagnóstico de Mantenimiento Predictivo")
    if rul_pred > 45:
        st.success(f"🟢 **ESTADO NOMINAL:** El motor #{selected_engine} posee un margen seguro de {rul_pred:.1f} ciclos remanentes. Operación continua aprobada.")
    elif 20 < rul_pred <= 45:
        st.warning(f"🟡 **ADVERTENCIA:** Vida útil remanente crítica ({rul_pred:.1f} ciclos). Se recomienda programar inspección boroscópica en el HPC.")
    else:
        st.error(f"🔴 **ALERTA MÁXIMA:** Retiro inmediato recomendado. RUL estimado de {rul_pred:.1f} ciclos cercano al umbral de falla funcional.")
        
    st.write("---")
    
    # Gráfico de Telemetría Dinámica
    st.subheader("📊 Histórico de Telemetría y Monitoreo de Sensores Críticos")
    sensor_options = [c for c in feature_cols if 'sensor' in c]
    sensor_to_plot = st.selectbox("Seleccionar Sensor para Inspeccionar Visualmente:", sensor_options if sensor_options else feature_cols)
    
    fig = px.line(engine_data, x=cycle_col, y=sensor_to_plot, title=f"Evolución de {sensor_to_plot} en Motor #{selected_engine}")
    fig.add_vline(x=selected_cycle, line_dash="dash", line_color="red", annotation_text="Ciclo Seleccionado")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Verificando disponibilidad de artefactos en data/processed/ y models/...")
