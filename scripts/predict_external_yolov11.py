from pathlib import Path
from collections import Counter
import random

from ultralytics import YOLO


# ============================================================
# SMARTROAD AI
# MODELO 3 — YOLO11n
# PRUEBA DE GENERALIZACIÓN EXTERNA
# ============================================================


# ------------------------------------------------------------
# 1. RUTA BASE DEL PROYECTO
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# 2. RUTAS
# ------------------------------------------------------------

EXTERNAL_IMAGES_DIR = PROJECT_DIR / "external_data" / "images"

MODEL_PATH = (
    PROJECT_DIR
    / "experiments"
    / "yolov11"
    / "entrenamiento_01_20_epochs"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "evaluation"
    / "yolov11"
    / "external_test"
    / "results"
)


# ------------------------------------------------------------
# 3. CONFIGURACIÓN
# ------------------------------------------------------------

IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU = 0.70
DEVICE = 0
SEED = 42


# ------------------------------------------------------------
# 4. CLASES DEL PROYECTO
# ------------------------------------------------------------

CLASS_NAMES = {
    0: "Accidente",
    1: "Bus",
    2: "Camion",
    3: "Carro",
    4: "Fire",
    5: "Moto",
}


# ------------------------------------------------------------
# 5. FUNCIÓN PRINCIPAL
# ------------------------------------------------------------

def main():
    print("=" * 70)
    print("SMARTROAD AI — MODELO 3 — YOLO11n")
    print("PRUEBA DE GENERALIZACIÓN EXTERNA")
    print("=" * 70)

    print("\nVerificando archivos...")

    if not EXTERNAL_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de imágenes externas:\n"
            f"{EXTERNAL_IMAGES_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n"
            f"{MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_files = list(OUTPUT_DIR.iterdir())

    if existing_files:
        raise FileExistsError(
            "\nLa carpeta de resultados ya contiene archivos:\n"
            f"{OUTPUT_DIR}\n\n"
            "No se sobrescribirán resultados existentes.\n"
            "Utiliza otra carpeta o vacía esta carpeta antes de ejecutar."
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
        ]
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No se encontraron imágenes en:\n"
            f"{EXTERNAL_IMAGES_DIR}"
        )

    print("\n" + "-" * 70)
    print("CONFIGURACIÓN")
    print("-" * 70)
    print("Modelo              : YOLO11n")
    print(f"Checkpoint           : {MODEL_PATH.name}")
    print(f"Imágenes externas    : {len(image_paths)}")
    print(f"Image size           : {IMAGE_SIZE}")
    print(f"Confianza            : {CONFIDENCE}")
    print(f"IoU                  : {IOU}")
    print(f"Device               : {DEVICE}")
    print(f"Seed                 : {SEED}")

    print("\nEntrada:")
    print(EXTERNAL_IMAGES_DIR)

    print("\nSalida:")
    print(OUTPUT_DIR)

    print("\n" + "-" * 70)
    print("IMÁGENES A PROCESAR")
    print("-" * 70)

    for index, image_path in enumerate(image_paths, start=1):
        print(f"{index:02d}. {image_path.name}")

    print("\n" + "=" * 70)
    print("CARGANDO YOLO11n")
    print("=" * 70)

    model = YOLO(str(MODEL_PATH))

    print("\n" + "=" * 70)
    print("GENERANDO PREDICCIONES")
    print("=" * 70)

    random.seed(SEED)

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
            "SMARTROAD AI — YOLO11n\n"
            "PRUEBA DE GENERALIZACIÓN EXTERNA\n"
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Modelo: YOLO11n\n"
            f"Checkpoint: {MODEL_PATH}\n"
            f"Imágenes procesadas: {processed_images}\n"
            f"Image size: {IMAGE_SIZE}\n"
            f"Confidence: {CONFIDENCE}\n"
            f"IoU: {IOU}\n"
            f"Device: {DEVICE}\n"
            f"Seed: {SEED}\n\n"
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
    print(f"\nImágenes procesadas : {processed_images}")
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
    print("Estas imágenes son externas al dataset oficial.")
    print(
        "La prueba evalúa generalización visual y no modifica "
        "el dataset original."
    )
    print(
        "Al no existir etiquetas Ground Truth para estas imágenes, "
        "esta prueba no calcula Precision, Recall ni mAP."
    )


# ------------------------------------------------------------
# EJECUCIÓN
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
