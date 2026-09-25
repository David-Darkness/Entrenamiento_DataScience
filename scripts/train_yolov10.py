from pathlib import Path
import tempfile
import yaml

from ultralytics import YOLO


# ============================================================
# SMARTROAD AI
# Modelo 2 — YOLOv10n
# Experimento 01 — 20 epochs
# ============================================================


# ------------------------------------------------------------
# 1. RUTA BASE DEL PROYECTO
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# 2. RUTAS DEL DATASET Y MODELO
# ------------------------------------------------------------

DATASET_DIR = BASE_DIR / "dataset"

ORIGINAL_DATA_YAML = DATASET_DIR / "data.yaml"

MODEL_PATH = BASE_DIR / "models" / "base" / "yolov10n.pt"


# ------------------------------------------------------------
# 3. DIRECTORIO DEL EXPERIMENTO
# ------------------------------------------------------------

PROJECT_DIR = BASE_DIR / "experiments" / "yolov10"

EXPERIMENT_NAME = "entrenamiento_01_20_epochs"

EXPERIMENT_DIR = PROJECT_DIR / EXPERIMENT_NAME


# ------------------------------------------------------------
# 4. FUNCIÓN PRINCIPAL
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("SMARTROAD AI — MODELO 2 — YOLOv10n")
    print("=" * 70)

    # --------------------------------------------------------
    # 5. VERIFICAR ARCHIVOS
    # --------------------------------------------------------

    print("\nVerificando archivos...")

    if not ORIGINAL_DATA_YAML.exists():
        raise FileNotFoundError(
            f"\nNo se encontró el dataset YAML:\n"
            f"{ORIGINAL_DATA_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nNo se encontró el modelo base YOLOv10n:\n"
            f"{MODEL_PATH}"
        )

    print("\nDataset YAML encontrado:")
    print(ORIGINAL_DATA_YAML)

    print("\nModelo base encontrado:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # 6. EVITAR SOBRESCRIBIR EL EXPERIMENTO
    # --------------------------------------------------------

    if EXPERIMENT_DIR.exists():
        raise FileExistsError(
            "\nEl directorio del experimento ya existe:\n"
            f"{EXPERIMENT_DIR}\n\n"
            "No se sobrescribirá el experimento existente."
        )

    # --------------------------------------------------------
    # 7. LEER data.yaml ORIGINAL
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("Cargando configuración del dataset...")
    print("-" * 70)

    with open(
        ORIGINAL_DATA_YAML,
        "r",
        encoding="utf-8"
    ) as file:

        data_config = yaml.safe_load(file)

    # --------------------------------------------------------
    # 8. DEFINIR RUTAS REALES DEL DATASET
    # --------------------------------------------------------

    train_images = (
        DATASET_DIR / "train" / "images"
    ).resolve()

    valid_images = (
        DATASET_DIR / "valid" / "images"
    ).resolve()

    test_images = (
        DATASET_DIR / "test" / "images"
    ).resolve()

    # --------------------------------------------------------
    # 9. VERIFICAR RUTAS
    # --------------------------------------------------------

    for name, path in {
        "Train": train_images,
        "Valid": valid_images,
        "Test": test_images,
    }.items():

        if not path.exists():
            raise FileNotFoundError(
                f"\nNo se encontró el directorio {name}:\n"
                f"{path}"
            )

    print("\nRutas utilizadas durante el entrenamiento:")

    print(f"  Train: {train_images}")
    print(f"  Valid: {valid_images}")
    print(f"  Test : {test_images}")

    # --------------------------------------------------------
    # 10. VERIFICAR CLASES
    # --------------------------------------------------------

    expected_names = [
        "Accidente",
        "Bus",
        "Camion",
        "Carro",
        "Fire",
        "Moto",
    ]

    if data_config.get("nc") != 6:
        raise ValueError(
            f"\nSe esperaban 6 clases, "
            f"pero data.yaml indica nc="
            f"{data_config.get('nc')}"
        )

    if data_config.get("names") != expected_names:
        raise ValueError(
            "\nLas clases del dataset no coinciden "
            "con la configuración esperada.\n\n"
            f"Esperadas: {expected_names}\n"
            f"Encontradas: {data_config.get('names')}"
        )

    print("\nClases del dataset:")

    for class_id, class_name in enumerate(
        expected_names
    ):
        print(f"  {class_id}: {class_name}")

    # --------------------------------------------------------
    # 11. CREAR YAML TEMPORAL
    # --------------------------------------------------------
    #
    # IMPORTANTE:
    #
    # dataset/data.yaml NO SE MODIFICA.
    #
    # Creamos un YAML temporal fuera del proyecto con
    # las rutas absolutas correctas.
    # --------------------------------------------------------

    runtime_data = {
        "train": str(train_images),
        "val": str(valid_images),
        "test": str(test_images),
        "nc": 6,
        "names": expected_names,
    }

    temp_yaml_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_smartroad_yolov10.yaml",
            prefix="data_",
            encoding="utf-8",
            delete=False,
        ) as temp_file:

            yaml.safe_dump(
                runtime_data,
                temp_file,
                sort_keys=False,
                allow_unicode=True,
            )

            temp_yaml_path = Path(
                temp_file.name
            )

        print("\nConfiguración temporal creada:")
        print(temp_yaml_path)

        # ----------------------------------------------------
        # 12. CARGAR YOLOv10n
        # ----------------------------------------------------

        print("\n" + "-" * 70)
        print("Cargando YOLOv10n...")
        print("-" * 70)

        model = YOLO(str(MODEL_PATH))

        # ----------------------------------------------------
        # 13. CONFIGURACIÓN DEL EXPERIMENTO
        # ----------------------------------------------------

        print("\nConfiguración experimental:")

        print("  Modelo       : YOLOv10n")
        print("  Epochs       : 20")
        print("  Image size   : 640")
        print("  Batch        : 2")
        print("  Device       : 0")
        print("  Workers      : 0")
        print("  Seed         : 42")
        print("  Dataset      : SmartRoad AI")
        print("  Clases       : 6")
        print("  Train        : 37.106 imágenes")
        print("  Valid        : 5.305 imágenes")
        print("  Test         : 2.648 imágenes")

        # ----------------------------------------------------
        # 14. INICIAR ENTRENAMIENTO
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("INICIANDO ENTRENAMIENTO")
        print("=" * 70)

        results = model.train(

            # YAML temporal con rutas correctas
            data=str(temp_yaml_path),

            # Número de épocas
            epochs=20,

            # Resolución
            imgsz=640,

            # Batch
            batch=2,

            # GPU
            device=0,

            # Workers
            workers=0,

            # Directorio de salida
            project=str(PROJECT_DIR),

            # Nombre del experimento
            name=EXPERIMENT_NAME,

            # No sobrescribir
            exist_ok=False,

            # Guardar pesos
            save=True,

            # Guardar checkpoint cada epoch
            save_period=1,

            # Validación durante entrenamiento
            val=True,

            # Semilla
            seed=42,

            # Mostrar información
            verbose=True,
        )

        # ----------------------------------------------------
        # 15. FINALIZACIÓN
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("ENTRENAMIENTO FINALIZADO")
        print("=" * 70)

        print("\nResultados guardados en:")
        print(EXPERIMENT_DIR)

        print("\nArchivos principales:")

        print(
            f"  best.pt     → "
            f"{EXPERIMENT_DIR / 'weights' / 'best.pt'}"
        )

        print(
            f"  last.pt     → "
            f"{EXPERIMENT_DIR / 'weights' / 'last.pt'}"
        )

        print(
            f"  results.csv → "
            f"{EXPERIMENT_DIR / 'results.csv'}"
        )

        print(
            f"  args.yaml   → "
            f"{EXPERIMENT_DIR / 'args.yaml'}"
        )

        print("\nEl Modelo 2 — YOLOv10n ha terminado.")
        print("El dataset original NO fue modificado.")
        print("El conjunto TEST permaneció reservado.")

    finally:

        # ----------------------------------------------------
        # 16. ELIMINAR YAML TEMPORAL
        # ----------------------------------------------------

        if (
            temp_yaml_path is not None
            and temp_yaml_path.exists()
        ):

            temp_yaml_path.unlink()

            print(
                "\nConfiguración temporal eliminada:"
            )

            print(temp_yaml_path)


# ------------------------------------------------------------
# 17. EJECUCIÓN SEGURA EN WINDOWS
# ------------------------------------------------------------

if __name__ == "__main__":
    main()