import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
import keras

BASE_DIR = Path(__file__).resolve().parent

WEIGHTS_PATH = BASE_DIR / "autoencoder_clean.weights.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"
THRESHOLD_PATH = BASE_DIR / "threshold.pkl"


def build_autoencoder(input_dim=78):
    inputs = keras.Input(shape=(input_dim,), name="input_layer")

    x = keras.layers.Dense(64, activation="relu", name="dense")(inputs)
    x = keras.layers.Dense(32, activation="relu", name="dense_1")(x)
    x = keras.layers.Dense(16, activation="relu", name="dense_2")(x)
    x = keras.layers.Dense(32, activation="relu", name="dense_3")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense_4")(x)

    outputs = keras.layers.Dense(input_dim, activation="sigmoid", name="dense_5")(x)

    model = keras.Model(inputs, outputs, name="functional")
    return model


@st.cache_resource
def load_assets():
    scaler = joblib.load(SCALER_PATH)

    threshold_raw = joblib.load(THRESHOLD_PATH)
    threshold = float(np.asarray(threshold_raw).ravel()[0])

    input_dim = scaler.n_features_in_

    model = build_autoencoder(input_dim)
    model.load_weights(str(WEIGHTS_PATH))

    return model, scaler, threshold


try:
    model, scaler, threshold = load_assets()
except Exception as e:
    st.error("Gagal load model / scaler / threshold.")
    st.exception(e)
    st.stop()


n_features = scaler.n_features_in_

st.title("🚀 Anomaly Detection (Autoencoder)")
st.write(f"Jumlah fitur harus: {n_features}")

# ==============================
# INPUT MANUAL
# ==============================
st.subheader("Input Manual")

input_data = st.text_input(f"Masukkan {n_features} angka, pisahkan dengan koma")

if st.button("Predict Manual"):
    try:
        data_list = list(map(float, input_data.split(",")))

        if len(data_list) != n_features:
            st.error(f"Jumlah fitur harus {n_features}")
        else:
            data = np.array([data_list])
            data_scaled = scaler.transform(data)

            recon = model.predict(data_scaled)
            mse = float(np.mean((data_scaled - recon) ** 2))

            pred = int(mse > threshold)

            if pred == 1:
                st.error(f"🚨 ANOMALY | MSE: {mse:.6f}")
            else:
                st.success(f"✅ NORMAL | MSE: {mse:.6f}")

    except Exception as e:
        st.warning(f"Error: {e}")


# ==============================
# UPLOAD CSV
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

            df["Prediction"] = pred
            df["MSE"] = mse

            st.write(df)
            st.success("✅ Prediksi selesai")

    except Exception as e:
        st.error(f"Error: {e}")
