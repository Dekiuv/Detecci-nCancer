import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

# ========================
# Configuración
# ========================
st.title("🧪 Clasificación de tumores de colon (MobileNetV2)")
st.write("Sube una imagen para predecir si el tumor es **benigno** o **maligno** usando un modelo entrenado con MobileNetV2.")

# Parámetros
img_size = (224, 224)

# Cargar modelo
model = load_model("modelo_mobilenet_colon.keras")

# ========================
# Procesamiento de imagen
# ========================
def procesar_imagen(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img)
    img_preprocessed = preprocess_input(img_array.astype(np.float32))
    return np.expand_dims(img_preprocessed, axis=0)  # (1, 224, 224, 3)

# ========================
# Subida de imagen
# ========================
uploaded_file = st.file_uploader("📤 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen subida", use_column_width=True)

    # Procesar y predecir
    imagen = procesar_imagen(uploaded_file)
    proba = model.predict(imagen)[0][0]
    pred = int(proba >= 0.5)

    clases = ['Benigno', 'Maligno']
    color = '🟩' if pred == 0 else '🟥'

    st.markdown(f"### Resultado: {color} **{clases[pred]}**")
    st.markdown(f"**Probabilidad de maligno:** {proba:.2%}")