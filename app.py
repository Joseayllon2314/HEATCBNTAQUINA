import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# Cargar archivos desde la misma carpeta
model = load_model("MODELOHEAT_V2.keras")
scaler = joblib.load("ESCALERHEAT_V2.pkl")

# Diccionario para los días
dias_semana = {
    "Lunes": 1, "Martes": 2, "Miércoles": 3, "Jueves": 4,
    "Viernes": 5, "Sábado": 6, "Domingo": 7
}

# === UI ===
st.set_page_config(page_title="ABINBEV - HEAT PREDICTION - TAQUINA", layout="centered")
st.image("TAQUINA.png", width=150)
st.title("ABINBEV - HEAT PREDICTION - TAQUINA")

st.markdown("---")
st.subheader("📥 Ingresar datos")

dia_str = st.selectbox("Día de la semana", list(dias_semana.keys()))
volumen_latas = st.number_input("Volumen de latas", min_value=0.0)
volumen_botellas = st.number_input("Volumen de botellas", min_value=0.0)

# Botón
if st.button("Predecir KPI"):
    if ((volumen_latas < 100 and volumen_botellas == 0) or
        (volumen_botellas < 100 and volumen_latas == 0)):
        st.error("❌ Uno de los volúmenes es menor a 100 y el otro es 0.")
    else:
        dia = dias_semana[dia_str]
        volumen_total = volumen_latas + volumen_botellas
        proporcion_latas = volumen_latas / volumen_total
        proporcion_botellas = volumen_botellas / volumen_total
        log_volumen_total = np.log1p(volumen_total)
        ratio_bot_lat = volumen_botellas / volumen_latas if volumen_latas > 0 else 0

        X_input = np.array([[dia, volumen_botellas, volumen_latas,
                            proporcion_latas, proporcion_botellas,
                            log_volumen_total, ratio_bot_lat]])

        X_scaled = scaler.transform(X_input)
        kpi_predicho = model.predict(X_scaled)[0][0]

        if kpi_predicho < 0:
            kpi_predicho = 0

        consumo_estimado = ((kpi_predicho * volumen_total) - 1033) / 34.26
        factor_correccion = 0.9671
        presion_distribucion = 1

        # Mostrar resultados
        st.markdown("### 📊 Resultados")
        st.write(f"**Volumen total:** {volumen_total:.2f}")
        st.write(f"**KPI PREDICHO:** {kpi_predicho:.4f}")
        st.write(f"**CONSUMO CALCULADO:** {consumo_estimado:.2f}")
        st.write(f"**FACTOR DE CORRECCIÓN:** {factor_correccion}")
        st.write(f"**PRESIÓN DE DISTRIBUCIÓN:** {presion_distribucion}")
