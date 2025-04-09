from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ========================
# PARÁMETROS
# ========================
img_size = (224, 224)
batch_size = 32
epochs = 10
ruta_imagenes = 'Image'  # estructura: Image/Benigno/, Image/Maligno/

# ========================
# GENERADORES DE DATOS
# ========================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    ruta_imagenes,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    ruta_imagenes,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# ========================
# MODELO CON MOBILENETV2
# ========================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # congelar capas base

# Capas superiores personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compilar modelo
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ========================
# ENTRENAMIENTO
# ========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# ========================
# GUARDAR MODELO
# ========================
model.save("modelo_mobilenet_colon.keras")
print("✅ Modelo guardado como modelo_mobilenet_colon.keras")

# ========================
# PLOT DE PRECISIÓN
# ========================
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión Validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()