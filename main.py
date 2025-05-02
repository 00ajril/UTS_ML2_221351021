import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="weather_temperature_prediction_model.tflite")
interpreter.allocate_tensors()

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Get input and output tensor info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# UI - Judul
st.title("🌤️ Prediksi Suhu Maksimum Harian di Seattle")

st.markdown("Masukkan data cuaca berikut untuk memprediksi suhu maksimum:")

# Input dari pengguna
precipitation = st.number_input("Presipitasi (mm)", min_value=0.0, max_value=100.0, value=0.1, step=0.1)
temp_min = st.number_input("Suhu Minimum (°C)", min_value=-10.0, max_value=40.0, value=5.0, step=0.1)
wind = st.number_input("Kecepatan Angin (m/s)", min_value=0.0, max_value=50.0, value=2.5, step=0.1)

weather_condition = st.selectbox("Kondisi Cuaca", ['sun', 'drizzle', 'fog', 'rain', 'snow'])

# One-hot encoding manual untuk kolom 'weather'
weather_columns = ['weather_drizzle', 'weather_fog', 'weather_rain', 'weather_snow']
weather_vector = [0, 0, 0, 0]

if weather_condition != 'sun':  # karena 'sun' adalah baseline (drop_first=True)
    col_index = weather_columns.index(f'weather_{weather_condition}')
    weather_vector[col_index] = 1

# Gabungkan semua fitur
input_data = pd.DataFrame({
    'precipitation': [precipitation],
    'temp_min': [temp_min],
    'wind': [wind],
    'weather_drizzle': [weather_vector[0]],
    'weather_fog': [weather_vector[1]],
    'weather_rain': [weather_vector[2]],
    'weather_snow': [weather_vector[3]]
})

# Normalisasi fitur numerik
numerical_features = ['precipitation', 'temp_min', 'wind']
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Ubah ke format LSTM (samples, timesteps, features)
input_array = input_data.astype(np.float32).values.reshape(1, input_data.shape[1], 1)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_array)

# Jalankan prediksi
interpreter.invoke()

# Ambil hasil prediksi
predicted_temp = interpreter.get_tensor(output_details[0]['index'])[0][0]

# Tampilkan hasil
st.subheader("🌡️ Hasil Prediksi:")
st.success(f"Suhu Maksimum yang Diprediksi: **{predicted_temp:.2f}°C**")
