# proyecto-cmapss-nasa
Pipeline de Mantenimiento Predictivo y Estimación de RUL para Motores Turbofán (NASA C-MAPSS FD001)
# Mantenimiento Predictivo & Estimación de RUL (NASA C-MAPSS FD001)

Este proyecto despliega un Dashboard interactivo de Mantenimiento Predictivo (PdM) enfocado en la estimación de la Vida Útil Remanente (RUL, *Remaining Useful Life*) de motores turbofán a partir de datos de telemetría de sensores, utilizando el conjunto de datos **NASA C-MAPSS (FD001)**.

**Aplicación en Vivo:** [Ver Dashboard en Streamlit Cloud](https://proyecto-cmapss-nasa-3ta9wqtadwukmwfbtmckuc.streamlit.app)

---

## Alcance del Proyecto

1. **Modelado Predictivo:** Algoritmo **XGBoost Regressor** entrenado con selección y reducción de características de 21 sensores y variables operacionales a las 17 señales de mayor significancia y varianza.
2. **Monitoreo en Tiempo Real:** Visualización dinámica de la degradación térmica (`sensor_2`) y de presión (`sensor_20`) a lo largo del ciclo de vida útil del motor.
3. **Alertas Tácticas de Mantenimiento:** Diagnóstico automatizado según el umbral de seguridad:
   * 🟢 **Estado Nominal:** RUL > 45 ciclos (Operación continua aprobada).
   * 🟡 **Advertencia:** 20 < RUL ≤ 45 ciclos (Inspección programada recomendada).
   * 🔴 **Alerta Máxima:** RUL ≤ 20 ciclos (Retiro inmediato y revisión funcional).

---

## Estructura del Repositorio

```text
.
├── data/
│   ├── processed/
│   │   └── train_processed_df.csv    # Conjunto de datos procesado y normalizado
│   └── raw/
│       ├── train_FD001.txt           # Datos crudos de entrenamiento C-MAPSS
│       ├── test_FD001.txt            # Datos crudos de prueba C-MAPSS
│       └── RUL_FD001.txt             # Valores reales de RUL objetivo
├── models/
│   ├── best_model_fd001.pkl          # Modelo XGBoost serializado
│   └── scaler_fd001.pkl              # Escalador normalizador (MinMaxScaler/StandardScaler)
├── notebooks/
│   └── pipeline_cmapss_fd001.ipynb   # Pipeline de EDA, preprocesamiento y entrenamiento
├── app.py                            # Código fuente de la interfaz Streamlit
├── requirements.txt                  # Dependencias del proyecto
└── README.md                         # Documentación del proyecto
