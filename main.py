import streamlit as st
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# ===== LOAD MODEL (pakai cache biar ringan) =====
@st.cache_resource
def load_assets():
    model = load_model("autoencoder_clean.keras", compile=False, safe_mode=False)
    scaler = joblib.load("scaler.pkl")
    threshold = joblib.load("threshold.pkl")
    return model, scaler, threshold

model, scaler, threshold = load_assets()

# ===== INFO FITUR =====
n_features = scaler.n_features_in_

# ===== UI =====
st.title("🚀 Anomaly Detection (Autoencoder)")
st.write(f"Jumlah fitur harus: {n_features}")

# ==============================
# 🔹 INPUT MANUAL
# ==============================
st.subheader("Input Manual")

input_data = st.text_input(f"Masukkan {n_features} angka (pisahkan koma)")

if st.button("Predict Manual"):
    try:
        data_list = list(map(float, input_data.split(",")))

        if len(data_list) != n_features:
            st.error(f"Jumlah fitur harus {n_features}")
        else:
            data = np.array([data_list])
            data_scaled = scaler.transform(data)

            recon = model.predict(data_scaled)
            mse = np.mean((data_scaled - recon) ** 2)

            pred = int(mse > threshold)

            if pred == 1:
                st.error(f"🚨 ANOMALY | MSE: {mse:.6f}")
            else:
                st.success(f"✅ NORMAL | MSE: {mse:.6f}")

    except Exception as e:
        st.warning(f"Error: {e}")

# ==============================
# 🔹 UPLOAD CSV
# ==============================
st.subheader("Upload CSV")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if df.shape[1] != n_features:
            st.error(f"Jumlah kolom harus {n_features}")
        else:
            X_scaled = scaler.transform(df)

            recon = model.predict(X_scaled)
            mse = np.mean((X_scaled - recon) ** 2, axis=1)

            pred = (mse > threshold).astype(int)

            df['Prediction'] = pred
            df['MSE'] = mse

            st.write(df)
            st.success("✅ Prediksi selesai")

    except Exception as e:
        st.error(f"Error: {e}")
