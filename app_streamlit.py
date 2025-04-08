import streamlit as st
import numpy as np
import joblib
from skimage.io import imread
from skimage.transform import resize
from PIL import Image

# Configuración
img_size = (64, 64)

# Cargar modelo
model = joblib.load('modelo_rf_imagenes.pkl')

# Título
st.title("🧪 Clasificación de tumores de colon")
st.write("Sube una imagen para predecir si el tumor es **benigno** o **maligno**.")

# Subida de imagen
uploaded_file = st.file_uploader("📤 Sube una imagen", type=["jpg", "jpeg", "png"])

def procesar_imagen(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize(img_size)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    return img_array.flatten()

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen subida", use_column_width=True)

    features = procesar_imagen(uploaded_file)
    features = features.reshape(1, -1)

    # Predecir
    proba = model.predict_proba(features)[0][1]
    umbral = 0.4
    pred = int(proba >= umbral)

    clases = ['Benigno', 'Maligno']
    color = '🟩' if pred == 0 else '🟥'

    st.markdown(f"### Resultado: {color} **{clases[pred]}** (Probabilidad de maligno: {proba:.2f}%)")