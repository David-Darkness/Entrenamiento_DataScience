from pathlib import Path
from collections import Counter

from ultralytics import RTDETR


# ============================================================
# SMARTROAD AI
# MODELO 4 - RT-DETR-L
# PRUEBA DE GENERALIZACION EXTERNA
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------
# RUTAS
# ------------------------------------------------------------

EXTERNAL_IMAGES_DIR = PROJECT_DIR / "external_data" / "images"

MODEL_PATH = (
    PROJECT_DIR
    / "experiments"
    / "rtdetr"
    / "entrenamiento_01_25_epochs"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "evaluation"
    / "rtdetr"
    / "external_test"
    / "results"
)

# ------------------------------------------------------------
# CONFIGURACION
# ------------------------------------------------------------

IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU = 0.70
DEVICE = 0

# ------------------------------------------------------------
# CLASES DEL PROYECTO
# ------------------------------------------------------------

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
    print("PRUEBA DE GENERALIZACION EXTERNA")
    print("=" * 70)

    print("\nVerificando archivos...")

    if not EXTERNAL_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de imagenes externas:\n"
            f"{EXTERNAL_IMAGES_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo:\n"
            f"{MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_files = list(OUTPUT_DIR.iterdir())

    if existing_files:
        raise FileExistsError(
            "\nLa carpeta de resultados ya contiene archivos:\n"
            f"{OUTPUT_DIR}\n\n"
            "No se sobrescribiran resultados existentes.\n"
            "Utiliza otra carpeta o vacia esta carpeta antes de ejecutar."
        )

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_paths = sorted(
        [
            path
            for path in EXTERNAL_IMAGES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in valid_extensions
        ],
        key=lambda path: path.name.lower(),
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No se encontraron imagenes en:\n"
            f"{EXTERNAL_IMAGES_DIR}"
        )

    print("\n" + "-" * 70)
    print("CONFIGURACION")
    print("-" * 70)
    print("Modelo              : RT-DETR-L")
    print(f"Checkpoint           : {MODEL_PATH.name}")
    print(f"Imagenes externas    : {len(image_paths)}")
    print(f"Image size           : {IMAGE_SIZE}")
    print(f"Confianza            : {CONFIDENCE}")
    print(f"IoU                  : {IOU}")
    print(f"Device               : {DEVICE}")

    print("\nEntrada:")
    print(EXTERNAL_IMAGES_DIR)

    print("\nSalida:")
    print(OUTPUT_DIR)

    print("\n" + "-" * 70)
    print("IMAGENES A PROCESAR")
    print("-" * 70)

    for index, image_path in enumerate(image_paths, start=1):
        print(f"{index:02d}. {image_path.name}")

    print("\n" + "=" * 70)
    print("CARGANDO RT-DETR-L")
    print("=" * 70)

    model = RTDETR(str(MODEL_PATH))

    print("\n" + "=" * 70)
    print("GENERANDO PREDICCIONES")
    print("=" * 70)

    total_detections = 0
    class_counter = Counter()
    processed_images = 0

    for image_path in image_paths:
        print(f"\nProcesando: {image_path.name}")

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=IOU,
            device=DEVICE,
            save=True,
            project=str(OUTPUT_DIR),
            name=".",
            exist_ok=True,
            verbose=False,
        )

        processed_images += 1

        for result in results:
            if result.boxes is None:
                continue

            num_detections = len(result.boxes)
            total_detections += num_detections

            for cls in result.boxes.cls.tolist():
                class_id = int(cls)
                class_name = CLASS_NAMES.get(
                    class_id,
                    f"Clase_{class_id}",
                )
                class_counter[class_name] += 1

            print(f"  Detecciones: {num_detections}")

    summary_path = OUTPUT_DIR / "summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(
            "SMARTROAD AI - RT-DETR-L\n"
            "PRUEBA DE GENERALIZACION EXTERNA\n"
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Modelo: RT-DETR-L\n"
            f"Checkpoint: {MODEL_PATH}\n"
            f"Imagenes procesadas: {processed_images}\n"
            f"Image size: {IMAGE_SIZE}\n"
            f"Confidence: {CONFIDENCE}\n"
            f"IoU: {IOU}\n"
            f"Device: {DEVICE}\n\n"
        )

        file.write(f"Detecciones totales: {total_detections}\n\n")
        file.write("DETECCIONES POR CLASE\n" + "-" * 40 + "\n")

        for class_name in CLASS_NAMES.values():
            file.write(
                f"{class_name}: "
                f"{class_counter.get(class_name, 0)}\n"
            )

    print("\n" + "=" * 70)
    print("PRUEBA EXTERNA COMPLETADA")
    print("=" * 70)
    print(f"\nImagenes procesadas : {processed_images}")
    print(f"Detecciones totales : {total_detections}")

    print("\nDetecciones por clase:")
    print("-" * 70)

    for class_name in CLASS_NAMES.values():
        print(
            f"{class_name:<12}: "
            f"{class_counter.get(class_name, 0)}"
        )

    print("\nResultados guardados en:")
    print(OUTPUT_DIR)
    print("\nResumen guardado en:")
    print(summary_path)

    print("\nIMPORTANTE:")
    print("Estas imagenes son externas al dataset oficial.")
    print(
        "La prueba evalua generalizacion visual y no modifica "
        "el dataset original."
    )
    print(
        "Al no existir etiquetas Ground Truth para estas imagenes, "
        "esta prueba no calcula Precision, Recall ni mAP."
    )


if __name__ == "__main__":
    main()
