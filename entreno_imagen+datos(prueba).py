import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate
from tensorflow.keras.models import Model

# ========== 1. CARGA DE DATOS ==========
print("📥 Cargando datos...")
df = pd.read_csv("Data/merged_dummies.csv")

# Variables clínicas (ajusta si tienes más útiles)
clinical_features = ['relapse', 'tumor_size', 'early_detection', 'sex', 'inflammatory_bowel_disease']
X_clinical = df[clinical_features]
y = df['survival_prediction']
image_names = df['image_name']

# ========== 2. ESCALADO DE VARIABLES CLÍNICAS ==========
print("⚙️ Escalando datos clínicos...")
scaler = StandardScaler()

X_clinical_scaled = scaler.fit_transform(X_clinical)

import joblib

joblib.dump(scaler, 'scaler_clinico.pkl')

# ========== 3. CARGA DE IMÁGENES CON SUBCARPETAS ==========
print("🖼️ Cargando imágenes desde subcarpetas...")

image_data = []
for name in image_names:
    # Detectar si es benigno o maligno por el nombre
    if name.lower().startswith("colonca"):  # Maligno
        path = f"Image/Maligno/{name}.jpeg"
    elif name.lower().startswith("colonn"):  # Benigno
        path = f"Image/Benigno/{name}.jpeg"
    else:
        print(f"⚠️ No se pudo clasificar: {name}")
        path = None

    try:
        img = load_img(path, target_size=(128, 128))
        img = img_to_array(img) / 255.0
        image_data.append(img)
    except:
        print(f"⚠️ Imagen no encontrada: {path}")
        image_data.append(np.zeros((128, 128, 3)))  # Imagen vacía como fallback

X_images = np.array(image_data)

# ========== 4. DIVISIÓN ENTRENAMIENTO/TEST ==========
print("🔀 Dividiendo conjuntos de datos...")
X_img_train, X_img_test, X_cli_train, X_cli_test, y_train, y_test = train_test_split(
    X_images, X_clinical_scaled, y, test_size=0.2, random_state=42
)

# ========== 5. MODELO HÍBRIDO ==========
print("🧠 Construyendo modelo híbrido...")

# Parte CNN (imágenes)
image_input = Input(shape=(128, 128, 3))
base_model = MobileNetV2(include_top=False, input_tensor=image_input, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Parte MLP (clínica)
clinical_input = Input(shape=(X_cli_train.shape[1],))
y_dense = Dense(64, activation='relu')(clinical_input)

# Fusión
combined = concatenate([x, y_dense])
z = Dense(64, activation='relu')(combined)
z = Dense(1, activation='sigmoid')(z)

# Modelo final
model = Model(inputs=[image_input, clinical_input], outputs=z)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ========== 6. ENTRENAMIENTO ==========
print("🚀 Entrenando modelo...")
model.fit(
    [X_img_train, X_cli_train], y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32
)

# ========== 7. EVALUACIÓN ==========
print("📊 Evaluando modelo en test...")
loss, accuracy = model.evaluate([X_img_test, X_cli_test], y_test)
print(f"\n🎯 Accuracy final del modelo híbrido: {accuracy:.2%}")

# ========== 8. GUARDAR EL MODELO ==========
print("💾 Guardando modelo entrenado...")
model.save("modelo_hibrido_cancer.h5")
print("✅ Modelo guardado como 'modelo_hibrido_cancer.h5'")
