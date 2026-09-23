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


# Carpetas de imágenes
train_images = DATASET_DIR / "train" / "images"
valid_images = DATASET_DIR / "valid" / "images"
test_images = DATASET_DIR / "test" / "images"

# Contar imágenes
train_count = len(list(train_images.iterdir()))
valid_count = len(list(valid_images.iterdir()))
test_count = len(list(test_images.iterdir()))

print("\nCantidad de imágenes:")
print(f"Train: {train_count}")
print(f"Valid: {valid_count}")
print(f"Test: {test_count}")