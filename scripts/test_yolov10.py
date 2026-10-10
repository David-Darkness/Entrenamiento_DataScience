from pathlib import Path
import tempfile
import yaml

from ultralytics import YOLO


# ================================================================
# SMARTROAD AI
# Modelo 2 — YOLOv10n
# Evaluación final sobre TEST
#
# IMPORTANTE:
# - No modifica dataset/
# - No genera archivos dentro de scripts/
# - Guarda únicamente los resultados en:
#   evaluation/yolov10/test_final/
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------
# RUTAS
# ----------------------------------------------------------------

DATASET_DIR = BASE_DIR / "dataset"
DATASET_YAML = DATASET_DIR / "data.yaml"

MODEL_PATH = (
    BASE_DIR
    / "experiments"
    / "yolov10"
    / "entrenamiento_01_20_epochs"
    / "weights"
    / "best.pt"
)

EVALUATION_ROOT = BASE_DIR / "evaluation" / "yolov10"
EVALUATION_DIR = EVALUATION_ROOT / "test_final"

# ----------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------

IMG_SIZE = 640
BATCH = 2
DEVICE = 0
WORKERS = 0
SEED = 42


def create_temp_yaml():
    """
    Crea un YAML temporal con rutas absolutas.

    El dataset/data.yaml original permanece intacto.
    """

    with open(DATASET_YAML, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    data["train"] = str((DATASET_DIR / "train" / "images").resolve())
    data["val"] = str((DATASET_DIR / "valid" / "images").resolve())
    data["test"] = str((DATASET_DIR / "test" / "images").resolve())

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_smartroad_yolov10_test.yaml",
        prefix="data_",
        delete=False,
        encoding="utf-8",
    )

    temp_path = Path(temp_file.name)

    with temp_file:
        yaml.safe_dump(
            data,
            temp_file,
            sort_keys=False,
            allow_unicode=True,
        )

    return temp_path


def main():

    print("=" * 70)
    print("SMARTROAD AI — MODELO 2 — YOLOv10n")
    print("EVALUACIÓN FINAL SOBRE TEST")
    print("=" * 70)

    # ============================================================
    # 1. VERIFICAR ARCHIVOS
    # ============================================================

    print("\nVerificando archivos...")

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{DATASET_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n{MODEL_PATH}"
        )

    test_images = DATASET_DIR / "test" / "images"

    if not test_images.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio TEST:\n{test_images}"
        )

    print(f"\nDataset YAML:")
    print(DATASET_YAML)

    print(f"\nModelo evaluado:")
    print(MODEL_PATH)

    print(f"\nTEST:")
    print(test_images)

    # ============================================================
    # 2. PROTEGER LA ESTRUCTURA
    # ============================================================

    if EVALUATION_DIR.exists():
        raise FileExistsError(
            "\nEl directorio de evaluación ya existe:\n"
            f"{EVALUATION_DIR}\n\n"
            "No se sobrescribirá una evaluación existente."
        )

    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 3. MOSTRAR CONFIGURACIÓN
    # ============================================================

    print("\n" + "-" * 70)
    print("CONFIGURACIÓN DE LA EVALUACIÓN")
    print("-" * 70)

    print("Modelo       : YOLOv10n")
    print("Checkpoint   : best.pt")
    print("Split        : TEST")
    print("Imágenes     : 2.648")
    print(f"Image size   : {IMG_SIZE}")
    print(f"Batch        : {BATCH}")
    print(f"Device       : {DEVICE}")
    print(f"Workers      : {WORKERS}")
    print(f"Seed         : {SEED}")

    print("\nSalida:")
    print(EVALUATION_DIR)

    # ============================================================
    # 4. CREAR YAML TEMPORAL
    # ============================================================

    temp_yaml = create_temp_yaml()

    print("\nConfiguración temporal creada:")
    print(temp_yaml)

    try:

        # ========================================================
        # 5. CARGAR YOLOv10n
        # ========================================================

        print("\n" + "=" * 70)
        print("CARGANDO YOLOv10n")
        print("=" * 70)

        model = YOLO(str(MODEL_PATH))

        # ========================================================
        # 6. EVALUAR ÚNICAMENTE SOBRE TEST
        # ========================================================

        print("\n" + "=" * 70)
        print("INICIANDO EVALUACIÓN SOBRE TEST")
        print("=" * 70)

        metrics = model.val(
            data=str(temp_yaml),
            split="test",
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            plots=True,
            save_json=False,
            verbose=True,
            project=str(EVALUATION_ROOT),
            name="test_final",
            exist_ok=False,
            seed=SEED,
        )

        # ========================================================
        # 7. RESULTADOS GLOBALES
        # ========================================================

        print("\n" + "=" * 70)
        print("RESULTADOS FINALES — YOLOv10n — TEST")
        print("=" * 70)

        print(f"Precision : {metrics.box.mp:.4f}")
        print(f"Recall    : {metrics.box.mr:.4f}")
        print(f"mAP@50    : {metrics.box.map50:.4f}")
        print(f"mAP@50-95 : {metrics.box.map:.4f}")

        # ========================================================
        # 8. RESULTADOS POR CLASE
        # ========================================================

        class_names = {
            0: "Accidente",
            1: "Bus",
            2: "Camion",
            3: "Carro",
            4: "Fire",
            5: "Moto",
        }

        print("\n" + "-" * 70)
        print("MÉTRICAS POR CLASE")
        print("-" * 70)

        # metrics.box.ap50 contiene AP@50 por clase
        # metrics.box.ap contiene AP@50-95 por clase
        # metrics.box.p / r contienen precision / recall por clase

        for class_id, class_name in class_names.items():

            try:
                precision = metrics.box.p[class_id]
                recall = metrics.box.r[class_id]
                ap50 = metrics.box.ap50[class_id]
                ap5095 = metrics.box.ap[class_id]

                print(
                    f"{class_id}: {class_name:<12} "
                    f"P={precision:.4f}  "
                    f"R={recall:.4f}  "
                    f"mAP50={ap50:.4f}  "
                    f"mAP50-95={ap5095:.4f}"
                )

            except (IndexError, TypeError):
                print(
                    f"{class_id}: {class_name:<12} "
                    "métricas por clase no disponibles"
                )

        # ========================================================
        # 9. UBICACIÓN FINAL
        # ========================================================

        print("\n" + "=" * 70)
        print("EVALUACIÓN COMPLETADA")
        print("=" * 70)

        print("\nResultados guardados en:")
        print(EVALUATION_DIR)

        print("\nEl dataset original NO fue modificado.")
        print("El split TEST fue utilizado únicamente para evaluación.")

    finally:

        # ========================================================
        # 10. ELIMINAR YAML TEMPORAL
        # ========================================================

        if temp_yaml.exists():
            temp_yaml.unlink()
            print("\nConfiguración temporal eliminada.")


if __name__ == "__main__":
    main()
