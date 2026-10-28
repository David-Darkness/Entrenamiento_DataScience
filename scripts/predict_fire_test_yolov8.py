from pathlib import Path
from ultralytics import YOLO
import random
import shutil

# ============================================================
# SMARTROAD AI - PRUEBA DIRIGIDA DE FIRE EN TEST
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "experiments"
    / "yolov8"
    / "entrenamiento_02_20_epochs_definitivo"
    / "weights"
    / "best.pt"
)

TEST_IMAGES_DIR = PROJECT_DIR / "dataset" / "test" / "images"
TEST_LABELS_DIR = PROJECT_DIR / "dataset" / "test" / "labels"

OUTPUT_DIR = (
    PROJECT_DIR
    / "evaluation"
    / "yolov8"
    / "test_final"
    / "predictions"
    / "fire_test"
)

# Cantidad de imágenes Fire a revisar
NUM_IMAGES = 20

# Confianza mínima para considerar una detección
CONFIDENCE = 0.25

# Semilla para que la selección sea reproducible
RANDOM_SEED = 42


# ============================================================
# CONFIGURACIÓN
# ============================================================

CLASS_NAMES = {
    0: "Accidente",
    1: "Bus",
    2: "Camion",
    3: "Carro",
    4: "Fire",
    5: "Moto",
}

FIRE_CLASS_ID = 4


# ============================================================
# BUSCAR IMÁGENES QUE REALMENTE CONTIENEN FIRE
# ============================================================

def find_fire_images():
    fire_images = []

    label_files = list(TEST_LABELS_DIR.glob("*.txt"))

    for label_file in label_files:

        try:
            lines = label_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        contains_fire = False

        for line in lines:
            parts = line.strip().split()

            if not parts:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            if class_id == FIRE_CLASS_ID:
                contains_fire = True
                break

        if contains_fire:

            # Buscar imagen correspondiente
            for extension in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:

                image_path = TEST_IMAGES_DIR / (label_file.stem + extension)

                if image_path.exists():
                    fire_images.append(image_path)
                    break

    return fire_images


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("SMARTROAD AI - PRUEBA DIRIGIDA DE FIRE EN TEST")
    print("=" * 65)

    print(f"\nModelo:")
    print(MODEL_PATH)

    print(f"\nDataset TEST:")
    print(TEST_IMAGES_DIR)

    print(f"\nEtiquetas TEST:")
    print(TEST_LABELS_DIR)

    print(f"\nSalida:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Verificar rutas
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        print("\nERROR: No se encontró el modelo.")
        return

    if not TEST_IMAGES_DIR.exists():
        print("\nERROR: No se encontró la carpeta de imágenes TEST.")
        return

    if not TEST_LABELS_DIR.exists():
        print("\nERROR: No se encontró la carpeta de labels TEST.")
        return

    # --------------------------------------------------------
    # Buscar imágenes Fire
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("BUSCANDO IMÁGENES DE FIRE EN TEST")
    print("=" * 65)

    fire_images = find_fire_images()

    print(f"\nImágenes TEST con Fire: {len(fire_images)}")

    if not fire_images:
        print("\nNo se encontraron imágenes con Fire.")
        return

    # --------------------------------------------------------
    # Selección reproducible
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    if len(fire_images) > NUM_IMAGES:
        selected_images = random.sample(fire_images, NUM_IMAGES)
    else:
        selected_images = fire_images

    print(f"Imágenes seleccionadas: {len(selected_images)}")

    # --------------------------------------------------------
    # Crear carpeta de salida
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Cargar modelo
    # --------------------------------------------------------

    print("\nCargando modelo...")

    model = YOLO(str(MODEL_PATH))

    print("Modelo cargado correctamente.")

    # --------------------------------------------------------
    # Predicciones
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("EJECUTANDO PREDICCIONES")
    print("=" * 65)

    fire_detected = 0
    fire_not_detected = 0

    total_fire_predictions = 0

    for image_path in selected_images:

        print("\n" + "-" * 65)
        print(f"Imagen: {image_path.name}")

        results = model.predict(
            source=str(image_path),
            conf=CONFIDENCE,
            save=True,
            project=str(OUTPUT_DIR),
            name="results",
            exist_ok=True,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = CLASS_NAMES.get(
                    class_id,
                    f"Clase_{class_id}"
                )

                detections.append(
                    (class_name, confidence)
                )

        # ----------------------------------------------------
        # Mostrar detecciones
        # ----------------------------------------------------

        if detections:

            print("Detecciones:")

            for class_name, confidence in detections:

                print(
                    f"  - {class_name} "
                    f"({confidence:.2f})"
                )

        else:

            print("  Sin detecciones.")

        # ----------------------------------------------------
        # Revisar Fire
        # ----------------------------------------------------

        fire_predictions = [
            confidence
            for class_name, confidence in detections
            if class_name == "Fire"
        ]

        if fire_predictions:

            fire_detected += 1
            total_fire_predictions += len(fire_predictions)

            print(
                f"  >>> FIRE DETECTADO "
                f"({len(fire_predictions)} detección/es)"
            )

        else:

            fire_not_detected += 1

            print(
                "  >>> FIRE NO DETECTADO "
                "(aunque la imagen contiene Fire)"
            )

    # ========================================================
    # RESUMEN
    # ========================================================

    print("\n")
    print("=" * 65)
    print("RESULTADO DE LA PRUEBA DIRIGIDA DE FIRE")
    print("=" * 65)

    print(f"\nImágenes TEST con Fire encontradas: {len(fire_images)}")
    print(f"Imágenes seleccionadas: {len(selected_images)}")

    print(f"\nImágenes donde YOLO detectó Fire: {fire_detected}")
    print(f"Imágenes donde YOLO NO detectó Fire: {fire_not_detected}")

    print(f"\nTotal de detecciones Fire: {total_fire_predictions}")

    if selected_images:

        detection_rate = (
            fire_detected / len(selected_images)
        ) * 100

        print(
            f"\nTasa de detección en esta muestra: "
            f"{detection_rate:.2f}%"
        )

    print("\nResultados guardados en:")

    print(OUTPUT_DIR / "results")

    print("\nProceso completado.")
    print("=" * 65)


if __name__ == "__main__":
    main()