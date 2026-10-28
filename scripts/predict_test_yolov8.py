from pathlib import Path
from ultralytics import YOLO
import random
import shutil


# ============================================================
# SMARTROAD AI - PREDICCIONES VISUALES SOBRE TEST
# Modelo 1: YOLOv8n
# ============================================================

# Raíz del proyecto
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Modelo definitivo
MODEL_PATH = (
    PROJECT_DIR
    / "experiments"
    / "yolov8"
    / "entrenamiento_02_20_epochs_definitivo"
    / "weights"
    / "best.pt"
)

# Imágenes originales del conjunto TEST
TEST_IMAGES_DIR = PROJECT_DIR / "dataset" / "test" / "images"

# Carpeta donde se guardarán las predicciones
OUTPUT_DIR = (
    PROJECT_DIR
    / "evaluation"
    / "yolov8"
    / "test_final"
    / "predictions"
)

# Cantidad de imágenes que vamos a revisar
NUM_IMAGES = 30

# Confianza mínima
CONFIDENCE = 0.25

# Semilla para que la selección sea reproducible
RANDOM_SEED = 42


# ============================================================
# VALIDACIONES
# ============================================================

print("=" * 60)
print("SMARTROAD AI - PREDICCIONES VISUALES YOLOv8")
print("=" * 60)

print(f"Modelo:")
print(MODEL_PATH)

print(f"\nDataset TEST:")
print(TEST_IMAGES_DIR)

print(f"\nSalida:")
print(OUTPUT_DIR)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nNo se encontró el modelo:\n{MODEL_PATH}"
    )

if not TEST_IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"\nNo se encontró la carpeta de imágenes TEST:\n{TEST_IMAGES_DIR}"
    )


# ============================================================
# PREPARAR CARPETA DE SALIDA
# ============================================================

GENERAL_DIR = OUTPUT_DIR / "muestras_generales"
FIRE_DIR = OUTPUT_DIR / "fire"

GENERAL_DIR.mkdir(parents=True, exist_ok=True)
FIRE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CARGAR MODELO
# ============================================================

print("\nCargando modelo...")

model = YOLO(str(MODEL_PATH))

print("Modelo cargado correctamente.")


# ============================================================
# OBTENER IMÁGENES TEST
# ============================================================

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p
    for p in TEST_IMAGES_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
]

print(f"\nImágenes TEST encontradas: {len(images)}")

if not images:
    raise RuntimeError("No se encontraron imágenes en TEST.")


# ============================================================
# SELECCIÓN REPRODUCIBLE
# ============================================================

random.seed(RANDOM_SEED)

sample_size = min(NUM_IMAGES, len(images))

selected_images = random.sample(images, sample_size)

print(f"Imágenes seleccionadas para revisión: {sample_size}")


# ============================================================
# PREDICCIÓN
# ============================================================

print("\nEjecutando predicciones...\n")

results = model.predict(
    source=[str(img) for img in selected_images],
    conf=CONFIDENCE,
    imgsz=640,
    device=0,
    save=False,
    verbose=False
)


# ============================================================
# PROCESAR RESULTADOS
# ============================================================

fire_found = 0
total_detections = 0

for image_path, result in zip(selected_images, results):

    # Nombre de salida
    output_path = GENERAL_DIR / image_path.name

    # Guardar imagen con bounding boxes
    plotted_image = result.plot()

    # Ultralytics devuelve BGR
    import cv2

    cv2.imwrite(
        str(output_path),
        plotted_image
    )

    # Analizar detecciones
    detected_classes = []

    if result.boxes is not None:

        for cls, conf in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist()
        ):

            class_id = int(cls)
            confidence = float(conf)

            class_name = result.names[class_id]

            detected_classes.append(
                f"{class_name} ({confidence:.2f})"
            )

            total_detections += 1

            if class_name.lower() == "fire":
                fire_found += 1

    print(f"\nImagen: {image_path.name}")

    if detected_classes:
        print("  Detecciones:")
        for detection in detected_classes:
            print(f"    - {detection}")
    else:
        print("  Sin detecciones.")


# ============================================================
# BUSCAR IMÁGENES CON FIRE
# ============================================================

print("\n" + "=" * 60)
print("BUSCANDO EJEMPLOS DE FIRE")
print("=" * 60)

for image_path, result in zip(selected_images, results):

    has_fire = False

    if result.boxes is not None:

        for cls in result.boxes.cls.tolist():

            class_id = int(cls)
            class_name = result.names[class_id]

            if class_name.lower() == "fire":
                has_fire = True
                break

    if has_fire:

        source_image = GENERAL_DIR / image_path.name
        destination_image = FIRE_DIR / image_path.name

        if source_image.exists():
            shutil.copy2(
                source_image,
                destination_image
            )


# ============================================================
# RESUMEN
# ============================================================

print("\n" + "=" * 60)
print("PREDICCIONES FINALIZADAS")
print("=" * 60)

print(f"Imágenes analizadas: {sample_size}")
print(f"Detecciones totales: {total_detections}")
print(f"Imágenes con Fire detectado: {fire_found}")

print("\nResultados guardados en:")

print(GENERAL_DIR)
print(FIRE_DIR)

print("\nProceso completado correctamente.")