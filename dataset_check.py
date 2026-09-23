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

# Contar objetos por clase
class_counts = {class_id: 0 for class_id in range(data["nc"])}

# Carpetas de etiquetas
train_labels = DATASET_DIR / "train" / "labels"
valid_labels = DATASET_DIR / "valid" / "labels"
test_labels = DATASET_DIR / "test" / "labels"

# Revisar las etiquetas de cada conjunto
for labels_folder in [train_labels, valid_labels, test_labels]:

    for label_file in labels_folder.glob("*.txt"):

        with open(label_file, "r", encoding="utf-8") as file:
            for line in file:

                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1


print("\nCantidad de objetos por clase:")

for class_id, count in class_counts.items():
    print(f"{class_id}: {data['names'][class_id]} → {count}")