import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Configuración
img_size = (64, 64)
benigno_dir = 'Dataset/Train/Train_Benigno'
maligno_dir = 'Dataset/Train/Train_Maligno'

def extract_features_from_image(path, size=img_size):
    try:
        img = imread(path)
        if img.ndim == 2:
            img = np.stack((img,)*3, axis=-1)
        img_resized = resize(img, size, anti_aliasing=True)
        return img_resized.flatten()
    except Exception as e:
        print(f"Error leyendo imagen {path}: {e}")
        return None

X, y = [], []

# Cargar benignos
for img_name in os.listdir(benigno_dir):
    path = os.path.join(benigno_dir, img_name)
    features = extract_features_from_image(path)
    if features is not None:
        X.append(features)
        y.append(0)

# Cargar malignos
for img_name in os.listdir(maligno_dir):
    path = os.path.join(maligno_dir, img_name)
    features = extract_features_from_image(path)
    if features is not None:
        X.append(features)
        y.append(1)

X = np.array(X)
y = np.array(y)

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calcular pesos de clase
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = {0: weights[0], 1: weights[1]}
print(f"Pesos usados: {class_weights}")

# Entrenar modelo con pesos
clf = RandomForestClassifier(n_estimators=200, class_weight=class_weights, random_state=42)
clf.fit(X_train, y_train)

# Predecir probabilidades
y_proba = clf.predict_proba(X_test)[:, 1]

# Ajustar umbral
umbral = 0.4
y_pred_ajustado = (y_proba >= umbral).astype(int)

# Evaluación
print("\nMatriz de confusión (umbral 0.4):")
print(confusion_matrix(y_test, y_pred_ajustado))

print("\nReporte de clasificación (umbral 0.4):")
print(classification_report(y_test, y_pred_ajustado))

import joblib

# Guardar el modelo
joblib.dump(clf, 'modelo_rf_imagenes.pkl')

# Guardar parámetros si lo necesitas (tamaño de imagen, umbral, etc.)
np.save('parametros.npy', np.array([64, 0.4]))  # ejemplo: tamaño, umbral