from pathlib import Path
import random

from ultralytics import YOLO


# ================================================================
# SMARTROAD AI
# Modelo 2 — YOLOv10n
# Prueba dirigida de la clase Fire sobre TEST
#
# IMPORTANTE:
# - NO modifica dataset/
# - NO genera salidas dentro de scripts/
# - Selecciona imágenes de TEST que realmente contienen Fire
#   según sus anotaciones.
# - Guarda las predicciones en:
#
#   evaluation/yolov10/test_final/
#       predictions/
#           fire_test/
#               results/
#
# Se seleccionan 20 imágenes, manteniendo el mismo tamaño de
# muestra utilizado en la prueba dirigida de Fire del Modelo 1.
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------
# RUTAS
# ----------------------------------------------------------------

TEST_IMAGES_DIR = BASE_DIR / "dataset" / "test" / "images"
TEST_LABELS_DIR = BASE_DIR / "dataset" / "test" / "labels"

MODEL_PATH = (
    BASE_DIR
    / "experiments"
    / "yolov10"
    / "entrenamiento_01_20_epochs"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "evaluation"
    / "yolov10"
    / "test_final"
    / "predictions"
    / "fire_test"
    / "results"
)

# ----------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------

FIRE_CLASS_ID = 4

NUM_SAMPLES = 20
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


def image_has_fire(label_path):
    """
    Devuelve True si el archivo YOLO contiene al menos una
    anotación de la clase Fire (class_id = 4).
    """

    if not label_path.exists():
        return False

    try:
        with open(label_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                # Formato YOLO esperado:
                # class_id x_center y_center width height
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue

                if class_id == FIRE_CLASS_ID:
                    return True

    except OSError:
        return False

    return False


def main():

    print("=" * 70)
    print("SMARTROAD AI — MODELO 2 — YOLOv10n")
    print("PRUEBA DIRIGIDA DE FIRE SOBRE TEST")
    print("=" * 70)

    # ============================================================
    # 1. VERIFICAR ARCHIVOS
    # ============================================================

    print("\nVerificando archivos...")

    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de imágenes TEST:\n"
            f"{TEST_IMAGES_DIR}"
        )

    if not TEST_LABELS_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de etiquetas TEST:\n"
            f"{TEST_LABELS_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n{MODEL_PATH}"
        )

    # ============================================================
    # 2. BUSCAR IMÁGENES QUE CONTIENEN FIRE
    # ============================================================

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    all_test_images = sorted(
        [
            path
            for path in TEST_IMAGES_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in image_extensions
        ],
        key=lambda p: p.name.lower(),
    )

    fire_images = []

    for image_path in all_test_images:

        label_path = TEST_LABELS_DIR / f"{image_path.stem}.txt"

        if image_has_fire(label_path):
            fire_images.append(image_path)

    print("\n" + "-" * 70)
    print("ANÁLISIS DEL TEST")
    print("-" * 70)

    print(f"Imágenes TEST totales          : {len(all_test_images)}")
    print(f"Imágenes TEST que contienen Fire: {len(fire_images)}")

    if len(fire_images) < NUM_SAMPLES:
        raise ValueError(
            f"Solamente se encontraron {len(fire_images)} imágenes "
            f"con Fire en TEST. Se necesitan {NUM_SAMPLES}."
        )

    # ============================================================
    # 3. SELECCIÓN REPRODUCIBLE
    # ============================================================

    rng = random.Random(SEED)

    selected_images = rng.sample(
        fire_images,
        NUM_SAMPLES,
    )

    selected_images.sort(
        key=lambda p: p.name.lower()
    )

    print(f"Muestras seleccionadas          : {len(selected_images)}")

    # ============================================================
    # 4. PROTEGER LA SALIDA
    # ============================================================

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "\nEl directorio de resultados ya existe:\n"
            f"{OUTPUT_DIR}\n\n"
            "No se sobrescribirá una prueba Fire existente."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # ============================================================
    # 5. CONFIGURACIÓN
    # ============================================================

    print("\n" + "-" * 70)
    print("CONFIGURACIÓN")
    print("-" * 70)

    print("Modelo       : YOLOv10n")
    print("Checkpoint   : best.pt")
    print("Split        : TEST")
    print("Clase        : Fire")
    print("Class ID     : 4")
    print(f"Muestras     : {NUM_SAMPLES}")
    print(f"Image size   : {IMG_SIZE}")
    print(f"Confianza    : {CONFIDENCE}")
    print(f"IoU          : {IOU}")
    print(f"Device       : {DEVICE}")
    print(f"Seed         : {SEED}")

    print("\nSalida:")
    print(OUTPUT_DIR)

    # ============================================================
    # 6. CARGAR MODELO
    # ============================================================

    print("\n" + "=" * 70)
    print("CARGANDO YOLOv10n")
    print("=" * 70)

    model = YOLO(str(MODEL_PATH))

    # ============================================================
    # 7. PREDICCIONES
    # ============================================================

    print("\n" + "=" * 70)
    print("GENERANDO PREDICCIONES DIRIGIDAS DE FIRE")
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
        project=str(
            BASE_DIR
            / "evaluation"
            / "yolov10"
            / "test_final"
            / "predictions"
            / "fire_test"
        ),
        name="results",
        exist_ok=False,
        verbose=False,
    )

    # ============================================================
    # 8. ANALIZAR LAS PREDICCIONES
    # ============================================================

    images_with_fire_prediction = 0
    total_fire_predictions = 0
    total_detections = 0

    detections_by_class = {
        name: 0
        for name in CLASS_NAMES.values()
    }

    for result in results:

        if result.boxes is None:
            continue

        image_has_fire_prediction = False

        total_detections += len(result.boxes)

        for class_id in result.boxes.cls.tolist():

            class_id = int(class_id)

            if class_id in CLASS_NAMES:

                class_name = CLASS_NAMES[class_id]
                detections_by_class[class_name] += 1

                if class_id == FIRE_CLASS_ID:
                    total_fire_predictions += 1
                    image_has_fire_prediction = True

        if image_has_fire_prediction:
            images_with_fire_prediction += 1

    # ============================================================
    # 9. RESULTADOS
    # ============================================================

    print("\n" + "=" * 70)
    print("PRUEBA DIRIGIDA FIRE COMPLETADA")
    print("=" * 70)

    print(f"\nImágenes Fire seleccionadas : {len(results)}")
    print(
        f"Imágenes con predicción Fire: "
        f"{images_with_fire_prediction}"
    )
    print(
        f"Imágenes sin predicción Fire: "
        f"{len(results) - images_with_fire_prediction}"
    )
    print(f"Predicciones Fire totales   : {total_fire_predictions}")
    print(f"Detecciones totales         : {total_detections}")

    print("\nDetecciones por clase:")
    print("-" * 70)

    for class_name, count in detections_by_class.items():
        print(f"{class_name:<12}: {count}")

    print("\nResultados guardados en:")
    print(OUTPUT_DIR)

    print("\nIMPORTANTE:")
    print(
        "Las 20 imágenes fueron seleccionadas porque sus etiquetas "
        "reales contienen al menos una instancia de Fire."
    )
    print("El dataset original NO fue modificado.")


if __name__ == "__main__":
    main()