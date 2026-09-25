from pathlib import Path
import yaml

# Ubicación del dataset
DATASET_DIR = Path("../dataset")

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



# Contar cuántas imágenes contienen cada clase
images_per_class = {class_id: 0 for class_id in range(data["nc"])}

for labels_folder in [train_labels, valid_labels, test_labels]:

    for label_file in labels_folder.glob("*.txt"):

        classes_in_image = set()

        with open(label_file, "r", encoding="utf-8") as file:
            for line in file:

                if line.strip():
                    class_id = int(line.split()[0])
                    classes_in_image.add(class_id)

        # Contamos la imagen una sola vez por cada clase
        for class_id in classes_in_image:
            images_per_class[class_id] += 1


print("\nCantidad de imágenes que contienen cada clase:")

for class_id, count in images_per_class.items():
    print(f"{class_id}: {data['names'][class_id]} → {count}")



print("\nDistribución de imágenes por clase y conjunto:")

splits = {
    "Train": train_labels,
    "Valid": valid_labels,
    "Test": test_labels
}

for split_name, labels_folder in splits.items():

    split_counts = {class_id: 0 for class_id in range(data["nc"])}

    for label_file in labels_folder.glob("*.txt"):

        classes_in_image = set()

        with open(label_file, "r", encoding="utf-8") as file:
            for line in file:

                if line.strip():
                    class_id = int(line.split()[0])
                    classes_in_image.add(class_id)

        for class_id in classes_in_image:
            split_counts[class_id] += 1

    print(f"\n{split_name}:")

    for class_id, count in split_counts.items():
        print(f"  {data['names'][class_id]} → {count}")

print("\n" + "=" * 50)
print("VALIDACIÓN DE IMÁGENES Y ETIQUETAS")
print("=" * 50)

splits_paths = {
    "Train": (
        DATASET_DIR / "train" / "images",
        DATASET_DIR / "train" / "labels"
    ),
    "Valid": (
        DATASET_DIR / "valid" / "images",
        DATASET_DIR / "valid" / "labels"
    ),
    "Test": (
        DATASET_DIR / "test" / "images",
        DATASET_DIR / "test" / "labels"
    )
}

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split_name, (images_folder, labels_folder) in splits_paths.items():

    images = {
        file.stem
        for file in images_folder.iterdir()
        if file.suffix.lower() in image_extensions
    }

    labels = {
        file.stem
        for file in labels_folder.glob("*.txt")
    }

    images_without_labels = images - labels
    labels_without_images = labels - images

    print(f"\n{split_name}:")
    print(f"  Imágenes: {len(images)}")
    print(f"  Etiquetas: {len(labels)}")
    print(f"  Imágenes sin etiqueta: {len(images_without_labels)}")
    print(f"  Etiquetas sin imagen: {len(labels_without_images)}")


print("\n" + "=" * 50)
print("VALIDACIÓN DEL CONTENIDO DE LAS ANOTACIONES")
print("=" * 50)

invalid_class_ids = []
invalid_coordinates = []
malformed_lines = []
empty_labels = []

for split_name, (_, labels_folder) in splits_paths.items():

    for label_file in labels_folder.glob("*.txt"):

        with open(label_file, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        # Archivo de etiquetas vacío
        if not lines:
            empty_labels.append((split_name, label_file.name))
            continue

        for line_number, line in enumerate(lines, start=1):

            parts = line.split()

            # Una anotación YOLO debe tener exactamente 5 valores
            if len(parts) != 5:
                malformed_lines.append(
                    (split_name, label_file.name, line_number, line)
                )
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                malformed_lines.append(
                    (split_name, label_file.name, line_number, line)
                )
                continue

            # Validar ID de clase
            if class_id < 0 or class_id >= data["nc"]:
                invalid_class_ids.append(
                    (split_name, label_file.name, line_number, class_id)
                )

            # Validar coordenadas
            coordinates = [x_center, y_center, width, height]

            if any(value < 0 or value > 1 for value in coordinates):
                invalid_coordinates.append(
                    (split_name, label_file.name, line_number, coordinates)
                )

print(f"\nArchivos de etiquetas vacíos: {len(empty_labels)}")

if empty_labels:
    print("\nArchivos vacíos encontrados:")
    for split_name, file_name in empty_labels:
        print(f"  [{split_name}] {file_name}")


print(f"\nLíneas con formato incorrecto: {len(malformed_lines)}")

if malformed_lines:
    print("\nLíneas incorrectas encontradas:")
    for split_name, file_name, line_number, line in malformed_lines:
        print(f"  [{split_name}] {file_name} | línea {line_number}")
        print(f"      Contenido: {line}")


print(f"\nIDs de clase inválidos: {len(invalid_class_ids)}")
print(f"Coordenadas fuera de rango: {len(invalid_coordinates)}")