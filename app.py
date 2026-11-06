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

# Carga segura de Artefactos de Modelo y Escalador
@st.cache_resource
def load_artifacts():
    try:
        model_path = os.path.join("models", "best_model_fd001.pkl")
        scaler_path = os.path.join("data", "processed", "scaler_fd001.pkl")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Error al cargar artefactos desde la ruta local: {e}")
        return None, None

@st.cache_data
def load_data():
    try:
        data_path = os.path.join("data", "processed", "train_processed_df.csv")
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"⚠️ Error al cargar el archivo procesado CSV: {e}")
        return None

# Cargar componentes principales
model, scaler = load_artifacts()
df = load_data()

if model is not None and df is not None:
    # Identificación dinámica de columnas de control
    id_col = next((col for col in ['unit_nr', 'engine_id', 'id'] if col in df.columns), df.columns[0])
    cycle_col = next((col for col in ['time_cycles', 'cycle', 'time'] if col in df.columns), df.columns[1])
    
    # Sidebar: Selección de Unidad y Ciclo
    st.sidebar.header("⚙️ Panel de Control de Inspección")
    engine_list = sorted(df[id_col].unique())
    selected_engine = st.sidebar.selectbox("Seleccionar ID de Motor:", engine_list)
    
    engine_data = df[df[id_col] == selected_engine].sort_values(cycle_col)
    max_cycle = int(engine_data[cycle_col].max())
    selected_cycle = st.sidebar.slider("Ciclo de Vuelo Actual:", min_value=1, max_value=max_cycle, value=max_cycle)
    
    # Fila de datos en el ciclo seleccionado
    current_row = engine_data[engine_data[cycle_col] == selected_cycle].copy()
    
    # Obtener las 21 columnas exactas que requiere el escalador (MinMaxScaler)
    if hasattr(scaler, "feature_names_in_"):
        expected_cols = list(scaler.feature_names_in_)
    else:
        # 3 Operational Settings + 21 Sensors = 24 total (21 features en FD001 tras descartes de varianza)
        expected_cols = [c for c in df.columns if c not in [id_col, cycle_col, 'RUL', 'Unnamed: 0']]
        if len(expected_cols) != getattr(scaler, 'n_features_in_', len(expected_cols)):
            # Construcción estándar C-MAPSS de 24 señales
            expected_cols = [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
            expected_cols = [c for c in expected_cols if c in df.columns]

    # Construir X_input asegurando el número exacto de características del Scaler (ej. 21)
    if hasattr(scaler, "n_features_in_") and len(expected_cols) != scaler.n_features_in_:
        # Filtrar solo columnas de sensores/settings existentes hasta igualar el n_features_in_
        candidates = [c for c in df.columns if 'sensor' in c.lower() or 'setting' in c.lower()]
        expected_cols = candidates[:scaler.n_features_in_]

    X_input = pd.DataFrame(index=[0])
    for col in expected_cols:
        if col in current_row.columns:
            val = current_row[col].values[0]
            X_input[col] = float(val) if pd.notnull(val) else 0.0
        else:
            X_input[col] = 0.0

    # Inferencia con manejo robusto
    try:
        raw_scaled = scaler.transform(X_input.values)
        
        # Intentar predecir con DataFrame o con Array según requiera el modelo
        try:
            if hasattr(model, "feature_names_in_"):
                X_scaled_df = pd.DataFrame(raw_scaled, columns=list(model.feature_names_in_))
                rul_pred = float(model.predict(X_scaled_df)[0])
            else:
                rul_pred = float(model.predict(raw_scaled)[0])
        except Exception:
            rul_pred = float(model.predict(raw_scaled)[0])
            
    except Exception as e_pred:
        st.error(f"Error procesando la inferencia del modelo: {e_pred}")
        rul_pred = 0.0
    
    # Visualización de Métricas Principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Motor Bajo Análisis", f"Unidad #{selected_engine}")
    col2.metric("Ciclo Evaluado", f"{selected_cycle} / {max_cycle}")
    col3.metric("RUL Estimado (XGBoost)", f"{rul_pred:.1f} ciclos")
    
    st.write("---")
    
    # Diagnóstico Táctico y Alertas de Mantenimiento (PdM)
    st.subheader("🔴 / 🟡 / 🟢 Diagnóstico de Mantenimiento Predictivo")
    if rul_pred > 45:
        st.success(f"🟢 **ESTADO NOMINAL:** El motor #{selected_engine} posee un margen seguro de {rul_pred:.1f} ciclos remanentes. Operación continua aprobada.")
    elif 20 < rul_pred <= 45:
        st.warning(f"🟡 **ADVERTENCIA:** Vida útil remanente crítica ({rul_pred:.1f} ciclos). Se recomienda programar inspección boroscópica en el HPC.")
    else:
        st.error(f"🔴 **ALERTA MÁXIMA:** Retiro inmediato recomendado. RUL estimado de {rul_pred:.1f} ciclos cercano al umbral de falla funcional.")
        
    st.write("---")
    
    # Gráfico Dinámico de Telemetría e Inspección Visual
    st.subheader("📊 Histórico de Telemetría y Monitoreo de Sensores Críticos")
    
    sensor_options = [c for c in df.columns if 'sensor' in c.lower()]
    if sensor_options:
        sensor_to_plot = st.selectbox("Seleccionar Sensor para Inspeccionar Visualmente:", sensor_options)
        
        fig = px.line(
            engine_data, 
            x=cycle_col, 
            y=sensor_to_plot, 
            title=f"Evolución de {sensor_to_plot} en Motor #{selected_engine}"
        )
        fig.add_vline(x=selected_cycle, line_dash="dash", line_color="red", annotation_text="Ciclo Seleccionado")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Verificando la disponibilidad de los artefactos en data/processed/ y models/...")
