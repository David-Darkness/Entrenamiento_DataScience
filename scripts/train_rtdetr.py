
from pathlib import Path

from ultralytics import RTDETR

# ============================================================
# SMARTROAD AI - Entrenamiento RT-DETR-L
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "rtdetr-l.pt"
MODEL_PATH = PROJECT_DIR / "models" / "base" / MODEL_NAME
DATASET_PATH = PROJECT_DIR / "dataset" / "data.yaml"

PROJECT_PATH = PROJECT_DIR / "experiments" / "rtdetr"
EXPERIMENT_NAME = "entrenamiento_01_25_epochs"
EXPERIMENT_DIR = PROJECT_PATH / EXPERIMENT_NAME


def ensure_model():
    """Descarga el modelo base si todavía no existe."""
    if MODEL_PATH.exists():
        print(f"Modelo base ya existe en: {MODEL_PATH}")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Descargando modelo base RT-DETR-L en: {MODEL_PATH}")

    # Ultralytics descarga los pesos oficiales si no existen localmente.
    model = RTDETR("rtdetr-l.pt")
    model.save(str(MODEL_PATH))

    print("Modelo base guardado correctamente.")


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
            "Cambia EXPERIMENT_NAME o mueve la carpeta anterior "
            "antes de iniciar un nuevo entrenamiento."
        )

    model = RTDETR(str(MODEL_PATH))

    print("=" * 70)
    print("SMARTROAD AI — Entrenamiento RT-DETR-L")
    print("=" * 70)
    print(f"Modelo base: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Salida: {EXPERIMENT_DIR}")
    print("Épocas: 25")
    print("Tamaño de imagen: 640")
    print("Batch: 2")
    print("=" * 70)

    model.train(
        data=str(DATASET_PATH),
        project=str(PROJECT_PATH),
        name=EXPERIMENT_NAME,
        epochs=25,
        imgsz=640,
        batch=2,
        device=0,
        workers=0,
        amp=True,
        val=True,
        save=True,
        save_period=1,
        patience=10,
        seed=0,
        deterministic=False,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO RT-DETR FINALIZADO")
    print("=" * 70)
    print(f"Resultados guardados en: {EXPERIMENT_DIR}")
    print("Archivos que debes revisar:")
    print("- args.yaml")
    print("- results.csv")
    print("- results.png")
    print("- confusion_matrix.png")
    print("- weights/best.pt")
    print("- weights/last.pt")


if __name__ == "__main__":
    main()
