from pathlib import Path
from urllib import request

from ultralytics import YOLO

# ============================================================
# SMARTROAD AI - Entrenamiento YOLO11n
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "yolo11n.pt"
MODEL_PATH = PROJECT_DIR / "models" / "base" / MODEL_NAME
DATASET_PATH = PROJECT_DIR / "dataset" / "data.yaml"
PROJECT_PATH = PROJECT_DIR / "experiments" / "yolov11"
EXPERIMENT_NAME = "entrenamiento_01_20_epochs"
EXPERIMENT_DIR = PROJECT_PATH / EXPERIMENT_NAME
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"


def ensure_model():
    if MODEL_PATH.exists():
        print(f"Modelo ya existe en: {MODEL_PATH}")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo base YOLO11n en: {MODEL_PATH}")
    print(f"URL: {MODEL_URL}")
    request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Descarga finalizada.")


def main():
    ensure_model()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset YAML:\n{DATASET_PATH}"
        )

    if EXPERIMENT_DIR.exists():
        raise FileExistsError(
            "El directorio del experimento ya existe:\n"
            f"{EXPERIMENT_DIR}\n\n"
            "Cambia el nombre del experimento o borra la carpeta anterior antes de entrenar."
        )

    model = YOLO(str(MODEL_PATH))

    print("=" * 70)
    print("SMARTROAD AI — Entrenamiento YOLO11n")
    print("=" * 70)
    print(f"Modelo base: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Salida: {EXPERIMENT_DIR}")

    model.train(
        data=str(DATASET_PATH),
        project=str(PROJECT_PATH),
        name=EXPERIMENT_NAME,
        epochs=20,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        amp=False,
        val=True,
        save=True,
        save_period=1,
        patience=10,
        seed=0,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO YOLO11 FINALIZADO")
    print("=" * 70)
    print(f"Resultados guardados en: {EXPERIMENT_DIR}")
    print("Archivos clave:")
    print("- args.yaml")
    print("- results.csv")
    print("- results.png")
    print("- confusion_matrix.png")
    print("- BoxP_curve.png")
    print("- BoxR_curve.png")
    print("- BoxPR_curve.png")
    print("- BoxF1_curve.png")
    print("- labels.jpg")
    print("- train_batch*.jpg")
    print("- val_batch*_labels.jpg")
    print("- val_batch*_pred.jpg")
    print("- weights/best.pt")
    print("- weights/last.pt")


if __name__ == "__main__":
    main()