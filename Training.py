import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, SimpleRNN, Dropout

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Lung Cancer Classification")

# Fungsi untuk memuat dataset
@st.cache_data(persist=True)
def load_data(filepath):
    return pd.read_csv(filepath)

# Fungsi untuk menampilkan distribusi target
def plot_target_distribution(df, target_column):
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.countplot(x=target_column, data=df, palette='pastel', ax=ax, hue=target_column)
    ax.set_title("Distribution of Target (LUNG_CANCER)")
    ax.set_xlabel("LUNG_CANCER (0 = No, 1 = Yes)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Fungsi untuk melakukan preprocessing data
def preprocess_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    # Encode categorical columns
    label_encoder = LabelEncoder()
    df['GENDER'] = label_encoder.fit_transform(df['GENDER'])
    df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])

    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_features = df.drop(columns=["LUNG_CANCER"]).columns
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

# Fungsi untuk membangun dan melatih model RNN
def build_and_train_rnn(X_train, y_train, input_shape):
    model = Sequential([
        Input(shape=(input_shape, 1)),
        SimpleRNN(32, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='tanh'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Reshape data untuk RNN
    X_train_rnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Train model
    history = model.fit(
        X_train_rnn, y_train, 
        epochs=50, batch_size=16, validation_split=0.2, verbose=0
    )
    return model, history

# Fungsi untuk evaluasi model
def evaluate_model(model, X_test, y_test):
    X_test_rnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_pred_prob = model.predict(X_test_rnn)
    y_pred = (y_pred_prob > 0.5).astype(int)

    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return conf_matrix, report

# Judul halaman
st.title('Lung Cancer Classification Using RNN')

# Memuat dataset
df = load_data('./survey lung cancer.csv')

# Menampilkan dataset
st.subheader("Dataset")
st.write(df.head())

st.subheader("Data Description")
st.write(df.describe())
st.write("Data Types")
st.write(df.dtypes)

# Visualisasi distribusi target
st.subheader("Target Distribution")
plot_target_distribution(df, target_column='LUNG_CANCER')

# Preprocessing data
st.subheader("Data Preprocessing")
st.write("Missing Values:", df.isnull().sum().sum())
st.write("Duplicated Values:", df.duplicated().sum())

df = preprocess_data(df)
st.write("Preprocessed Data")
st.write(df.head())

# Split data
st.subheader("Data Splitting")
X = df.drop(columns=["LUNG_CANCER"])
y = df["LUNG_CANCER"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
st.write(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# Model training
st.subheader("Model Training")
model, history = build_and_train_rnn(X_train, y_train, X_train.shape[1])

# Evaluasi model
st.subheader("Model Evaluation")
conf_matrix, report = evaluate_model(model, X_test, y_test)

# Classification Report
st.write("### Classification Report")
st.write(pd.DataFrame(report).transpose())

# Confusion Matrix
st.write("### Confusion Matrix")
fig, ax = plt.subplots(figsize=(3, 3))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
disp.plot(ax=ax)
st.pyplot(fig)
