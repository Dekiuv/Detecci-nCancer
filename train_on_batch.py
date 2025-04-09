from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt

# Cargar el modelo guardado previamente
modelo_hibrido = load_model("modelo_hibrido_inicial.keras")

# Configuración
ruta_lotes = "D:\Fotos"
batch_size = 128  # <--- antes estaba en 500
n_lotes = len([f for f in os.listdir(ruta_lotes) if f.startswith("X_img_batch_")])
epochs = 30

# Métricas por época
history_loss = []
history_acc = []

# Entrenamiento por lotes
for epoch in range(epochs):
    epoch_loss = 0
    epoch_acc = 0

    print(f"\n🌀 Época {epoch+1}/{epochs}")
    for i in range(1, n_lotes + 1):
        # Cargar lote
        X_img = np.load(os.path.join(ruta_lotes, f"X_img_batch_{i}.npy"))
        X_tab = np.load(os.path.join(ruta_lotes, f"X_tab_batch_{i}.npy"))
        y = np.load(os.path.join(ruta_lotes, f"y_batch_{i}.npy"))

        # Entrenar sobre lote
        loss, acc = modelo_hibrido.train_on_batch([X_img, X_tab], y)
        epoch_loss += loss
        epoch_acc += acc

        print(f"  🔹 Lote {i}/{n_lotes} → loss: {loss:.4f} | acc: {acc:.4f}")

    # Promedios por época
    avg_loss = epoch_loss / n_lotes
    avg_acc = epoch_acc / n_lotes

    history_loss.append(avg_loss)
    history_acc.append(avg_acc)

    print(f"📈 Promedio época {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

# Guardar modelo final
modelo_hibrido.save("modelo_hibrido_entrenado.keras")
print("✅ Modelo final guardado como modelo_hibrido_entrenado.keras")

# Graficar métricas
plt.figure(figsize=(10, 5))
plt.plot(history_loss, label="Pérdida (loss)")
plt.plot(history_acc, label="Precisión (accuracy)")
plt.title("Evolución del entrenamiento")
plt.xlabel("Época")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("entrenamiento_hibrido.png")
plt.show()