from pathlib import Path
import yaml

# Ubicación del dataset
DATASET_DIR = Path("dataset")

# Archivo de configuración
DATA_YAML = DATASET_DIR / "data.yaml"

# Leer el archivo YAML
with open(DATA_YAML, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

# Mostrar las clases
print("Clases del dataset:")
for class_id, class_name in enumerate(data["names"]):
    print(f"{class_id}: {class_name}")


