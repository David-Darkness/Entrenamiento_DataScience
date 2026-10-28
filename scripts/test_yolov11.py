from pathlib import Path
import tempfile
import yaml

from ultralytics import YOLO


# ================================================================
# SMARTROAD AI
# Modelo 3 - YOLO11n
# Evaluacion final sobre TEST
#
# IMPORTANTE:
# - No modifica dataset/
# - No genera archivos dentro de scripts/
# - Guarda unicamente los resultados en:
#   evaluation/yolov11/test_final/
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
    / "yolov11"
    / "entrenamiento_01_20_epochs"
    / "weights"
    / "best.pt"
)

EVALUATION_ROOT = BASE_DIR / "evaluation" / "yolov11"
EVALUATION_DIR = EVALUATION_ROOT / "test_final"

# ----------------------------------------------------------------
# CONFIGURACION
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
        suffix="_smartroad_yolov11_test.yaml",
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
    print("SMARTROAD AI - MODELO 3 - YOLO11n")
    print("EVALUACION FINAL SOBRE TEST")
    print("=" * 70)

    print("\nVerificando archivos...")

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"No se encontro:\n{DATASET_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo:\n{MODEL_PATH}"
        )

    test_images = DATASET_DIR / "test" / "images"

    if not test_images.exists():
        raise FileNotFoundError(
            f"No se encontro el directorio TEST:\n{test_images}"
        )

    print(f"\nDataset YAML:")
    print(DATASET_YAML)

    print(f"\nModelo evaluado:")
    print(MODEL_PATH)

    print(f"\nTEST:")
    print(test_images)

    if EVALUATION_DIR.exists():
        raise FileExistsError(
            "\nEl directorio de evaluacion ya existe:\n"
            f"{EVALUATION_DIR}\n\n"
            "No se sobrescribira una evaluacion existente."
        )

    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 70)
    print("CONFIGURACION DE LA EVALUACION")
    print("-" * 70)
    print("Modelo       : YOLO11n")
    print("Checkpoint   : best.pt")
    print("Split        : TEST")
    print("Imagenes     : 2.648")
    print(f"Image size   : {IMG_SIZE}")
    print(f"Batch        : {BATCH}")
    print(f"Device       : {DEVICE}")
    print(f"Workers      : {WORKERS}")
    print(f"Seed         : {SEED}")
    print("\nSalida:")
    print(EVALUATION_DIR)

    temp_yaml = create_temp_yaml()
    print("\nConfiguracion temporal creada:")
    print(temp_yaml)

    try:
        print("\n" + "=" * 70)
        print("CARGANDO YOLO11n")
        print("=" * 70)

        model = YOLO(str(MODEL_PATH))

        print("\n" + "=" * 70)
        print("INICIANDO EVALUACION SOBRE TEST")
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

        print("\n" + "=" * 70)
        print("RESULTADOS FINALES - YOLO11n - TEST")
        print("=" * 70)

        print(f"Precision : {metrics.box.mp:.4f}")
        print(f"Recall    : {metrics.box.mr:.4f}")
        print(f"mAP@50    : {metrics.box.map50:.4f}")
        print(f"mAP@50-95 : {metrics.box.map:.4f}")

        class_names = {
            0: "Accidente",
            1: "Bus",
            2: "Camion",
            3: "Carro",
            4: "Fire",
            5: "Moto",
        }

        print("\n" + "-" * 70)
        print("METRICAS POR CLASE")
        print("-" * 70)

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
                    "metricas por clase no disponibles"
                )

        print("\n" + "=" * 70)
        print("EVALUACION COMPLETADA")
        print("=" * 70)
        print("\nResultados guardados en:")
        print(EVALUATION_DIR)
        print("\nEl dataset original NO fue modificado.")
        print("El split TEST fue utilizado unicamente para evaluacion.")

    finally:
        if temp_yaml.exists():
            temp_yaml.unlink()
            print("\nConfiguracion temporal eliminada.")


if __name__ == "__main__":
    main()
