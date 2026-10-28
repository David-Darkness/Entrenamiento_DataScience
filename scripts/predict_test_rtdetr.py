from pathlib import Path
import random

from ultralytics import RTDETR


# ================================================================
# SMARTROAD AI
# Modelo 4 - RT-DETR-L
# Prediccion visual general sobre TEST
#
# IMPORTANTE:
# - NO modifica dataset/
# - NO genera salidas dentro de scripts/
# - Utiliza unicamente el best.pt del entrenamiento definitivo
# - Guarda las imagenes anotadas en:
#   evaluation/rtdetr/test_final/predictions/muestras_generales/
#
# Se generan 30 muestras con una seleccion reproducible.
# ================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------
# RUTAS
# ----------------------------------------------------------------

DATASET_TEST = PROJECT_DIR / "dataset" / "test" / "images"

MODEL_PATH = (
    PROJECT_DIR
    / "experiments"
    / "rtdetr"
    / "entrenamiento_01_25_epochs"
    / "weights"
    / "best.pt"
)

EVALUATION_DIR = PROJECT_DIR / "evaluation" / "rtdetr" / "test_final"
OUTPUT_DIR = EVALUATION_DIR / "predictions" / "muestras_generales"

# ----------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------

NUM_SAMPLES = 30
IMG_SIZE = 640
CONFIDENCE = 0.25
IOU = 0.70
DEVICE = 0
SEED = 42

CLASS_NAMES = {
    0: "Accidente",
    1: "Bus",
    2: "Camion",
    3: "Carro",
    4: "Fire",
    5: "Moto",
}


def main():
    print("=" * 70)
    print("SMARTROAD AI - MODELO 4 - RT-DETR-L")
    print("PREDICCION VISUAL GENERAL SOBRE TEST")
    print("=" * 70)

    print("\nVerificando archivos...")

    if not DATASET_TEST.exists():
        raise FileNotFoundError(
            f"No se encontro el directorio TEST:\n{DATASET_TEST}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo:\n{MODEL_PATH}"
        )

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = sorted(
        [
            path
            for path in DATASET_TEST.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        ],
        key=lambda path: path.name.lower(),
    )

    if len(all_images) < NUM_SAMPLES:
        raise ValueError(
            f"TEST contiene solamente {len(all_images)} imagenes. "
            f"Se necesitan al menos {NUM_SAMPLES}."
        )

    rng = random.Random(SEED)
    selected_images = rng.sample(all_images, NUM_SAMPLES)

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "\nEl directorio de predicciones ya existe:\n"
            f"{OUTPUT_DIR}\n\n"
            "No se sobrescribira una evaluacion visual existente."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\n" + "-" * 70)
    print("CONFIGURACION")
    print("-" * 70)
    print("Modelo       : RT-DETR-L")
    print("Checkpoint   : best.pt")
    print("Split        : TEST")
    print(f"Imagenes TEST: {len(all_images)}")
    print(f"Muestras     : {NUM_SAMPLES}")
    print(f"Image size   : {IMG_SIZE}")
    print(f"Confianza    : {CONFIDENCE}")
    print(f"IoU          : {IOU}")
    print(f"Device       : {DEVICE}")
    print(f"Seed         : {SEED}")
    print("\nSalida:")
    print(OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("CARGANDO RT-DETR-L")
    print("=" * 70)

    model = RTDETR(str(MODEL_PATH))

    print("\n" + "=" * 70)
    print("GENERANDO PREDICCIONES")
    print("=" * 70)

    results = model.predict(
        source=[str(path) for path in selected_images],
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        iou=IOU,
        device=DEVICE,
        save=True,
        save_txt=False,
        save_conf=False,
        show_labels=True,
        show_conf=True,
        line_width=2,
        project=str(EVALUATION_DIR / "predictions"),
        name="muestras_generales",
        exist_ok=False,
        verbose=False,
    )

    total_detections = 0
    detections_by_class = {name: 0 for name in CLASS_NAMES.values()}

    for result in results:
        if result.boxes is None:
            continue

        total_detections += len(result.boxes)

        for class_id in result.boxes.cls.tolist():
            class_id = int(class_id)

            if class_id in CLASS_NAMES:
                detections_by_class[CLASS_NAMES[class_id]] += 1

    print("\n" + "=" * 70)
    print("PREDICCION VISUAL COMPLETADA")
    print("=" * 70)
    print(f"\nImagenes procesadas : {len(results)}")
    print(f"Detecciones totales : {total_detections}")
    print("\nDetecciones por clase:")
    print("-" * 70)

    for class_name, count in detections_by_class.items():
        print(f"{class_name:<12}: {count}")

    print("\nResultados guardados en:")
    print(OUTPUT_DIR)
    print("\nEl dataset original NO fue modificado.")
    print("Las predicciones se generaron unicamente sobre imagenes de TEST.")


if __name__ == "__main__":
    main()
