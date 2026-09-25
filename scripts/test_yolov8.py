from pathlib import Path
from ultralytics import YOLO

# ============================================================
# SMARTROAD AI - Evaluación del modelo YOLOv8
# ============================================================

# Ruta raíz del proyecto
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Modelo entrenado
MODEL_PATH = PROJECT_DIR / "experiments" / "yolov8" / "entrenamiento_02_20_epochs_definitivo" / "weights" / "best.pt"

# Dataset
DATA_PATH = PROJECT_DIR / "dataset" / "data.yaml"

print("==============================================")
print("SMARTROAD AI - EVALUACIÓN YOLOv8")
print("==============================================")

print(f"Modelo: {MODEL_PATH}")
print(f"Dataset: {DATA_PATH}")

# Verificar archivos
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el dataset: {DATA_PATH}")

# Cargar modelo
model = YOLO(str(MODEL_PATH))

# Evaluar sobre TEST
results = model.val(
    data=str(DATA_PATH),
    split="test",
    imgsz=640,
    batch=2,
    device=0,
    workers=0,
    plots=True,
    verbose=True
)

print("\n==============================================")
print("EVALUACIÓN FINALIZADA")
print("==============================================")