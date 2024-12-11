import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Lung Cancer Prediction")

# Memuat model
@st.cache_resource
def load_rnn_model(model_path):
    return load_model(model_path)

# Memuat model
model_path = './model/rnn_lung_cancer_model.keras'  # Path model RNN
model = load_rnn_model(model_path)

# Fungsi preprocessing data
def preprocess_data(inputs):
    # Normalisasi age
    minimal, maximal = 21, 87
    if inputs[0] < minimal:
        inputs[0] = minimal
    elif inputs[0] > maximal:
        inputs[0] = maximal
    
    inputs[0] = inputs[0] - minimal / maximal - minimal  # Normalisasi age ke rentang 0-1
    return inputs

# Fungsi prediksi
def predict(inputs):
    inputs_array = np.array(inputs).reshape(1, -1, 1)
    probability = model.predict(inputs_array)[0][0]
    prediction = 1 if probability > 0.5 else 0
    return probability, prediction

# Judul aplikasi
st.title("Lung Cancer Prediction")
st.write("Masukkan nilai untuk setiap fitur untuk memprediksi kemungkinan kanker paru-paru.")

# Input fitur dari pengguna
age = st.slider("Age", min_value=18, max_value=100, value=50, step=1)

# Ubah gender selectbox menjadi Male dan Female
gender = st.selectbox("Gender", options=["Female", "Male"])
gender = 0 if gender == "Female" else 1  # Mappings: Female -> 0, Male -> 1

# Mengubah fitur lainnya menjadi biner (0 = No, 1 = Yes)
smoking = st.selectbox("Smoking", options=["No", "Yes"])
smoking = 1 if smoking == "Yes" else 0

yellow_fingers = st.selectbox("Yellow Fingers", options=["No", "Yes"])
yellow_fingers = 1 if yellow_fingers == "Yes" else 0

anxiety = st.selectbox("Anxiety", options=["No", "Yes"])
anxiety = 1 if anxiety == "Yes" else 0

peer_pressure = st.selectbox("Peer Pressure", options=["No", "Yes"])
peer_pressure = 1 if peer_pressure == "Yes" else 0

chronic_disease = st.selectbox("Chronic Disease", options=["No", "Yes"])
chronic_disease = 1 if chronic_disease == "Yes" else 0

fatigue = st.selectbox("Fatigue", options=["No", "Yes"])
fatigue = 1 if fatigue == "Yes" else 0

allergy = st.selectbox("Allergy", options=["No", "Yes"])
allergy = 1 if allergy == "Yes" else 0

wheezing = st.selectbox("Wheezing", options=["No", "Yes"])
wheezing = 1 if wheezing == "Yes" else 0

alcohol_consuming = st.selectbox("Alcohol Consuming", options=["No", "Yes"])
alcohol_consuming = 1 if alcohol_consuming == "Yes" else 0

coughing = st.selectbox("Coughing", options=["No", "Yes"])
coughing = 1 if coughing == "Yes" else 0

shortness_of_breath = st.selectbox("Shortness of Breath", options=["No", "Yes"])
shortness_of_breath = 1 if shortness_of_breath == "Yes" else 0

swallowing_difficulty = st.selectbox("Swallowing Difficulty", options=["No", "Yes"])
swallowing_difficulty = 1 if swallowing_difficulty == "Yes" else 0

chest_pain = st.selectbox("Chest Pain", options=["No", "Yes"])
chest_pain = 1 if chest_pain == "Yes" else 0

# Buat array input
inputs = [
    age, gender, smoking, yellow_fingers, anxiety, peer_pressure,
    chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
    coughing, shortness_of_breath, swallowing_difficulty, chest_pain
]


# Tombol prediksi
if st.button("Predict"):
    preprocessed_inputs = preprocess_data(inputs)
    probability, prediction = predict(preprocessed_inputs)
    st.write(preprocessed_inputs)
    st.write(f"Predicted Probability: {probability:.2f}")
    st.write(f"Prediction: {'Positive for Lung Cancer' if prediction == 1 else 'Negative for Lung Cancer'}")
