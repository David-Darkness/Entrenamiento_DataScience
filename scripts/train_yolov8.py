from ultralytics import YOLO

# ============================================================
# SMARTROAD AI - Entrenamiento YOLOv8n
# ============================================================

# Modelo base preentrenado
model = YOLO("../yolov8n.pt")

# Entrenamiento
results = model.train(
    data="../dataset/data.yaml",

    # Entrenamiento
    epochs=20,
    imgsz=640,
    batch=2,

    # GPU
    device=0,

    # Windows
    workers=0,

    # Optimizaciones
    amp=True,

    # Validación durante el entrenamiento
    val=True,

    # Guardar resultados
    save=True,
    save_period=1,

    # Nombre del experimento
    project="../runs/smartroad",
    name="yolov8n_20epochs",

    # Early stopping
    patience=10,

    # Reproducibilidad
    seed=0,

    # Generar gráficas
    plots=True,

    # Mostrar progreso detallado
    verbose=True
)

print("\n" + "=" * 60)
print("ENTRENAMIENTO FINALIZADO")
print("=" * 60)

print("Resultados guardados en:")
print("../runs/smartroad/yolov8n_20epochs")

print("\nArchivos importantes generados por Ultralytics:")
print("- results.csv")
print("- results.png")
print("- confusion_matrix.png")
print("- confusion_matrix_normalized.png")
print("- BoxF1_curve.png")
print("- BoxP_curve.png")
print("- BoxPR_curve.png")
print("- BoxR_curve.png")
print("- labels.jpg")
print("- train_batch*.jpg")
print("- val_batch*_labels.jpg")
print("- val_batch*_pred.jpg")

print("\nPesos:")
print("- weights/best.pt")
print("- weights/last.pt")

print("\nEntrenamiento completado.")
