from pathlib import Path
import shutil

DATASET_DIR = Path("dataset")

labels_folders = [
    DATASET_DIR / "train" / "labels",
    DATASET_DIR / "valid" / "labels",
    DATASET_DIR / "test" / "labels"
]

# Carpeta para guardar copias de seguridad
BACKUP_DIR = DATASET_DIR / "labels_backup"

converted = 0

print("=" * 50)
print("CONVERSIÓN DE POLÍGONOS A BOUNDING BOX")
print("=" * 50)

for labels_folder in labels_folders:

    for label_file in labels_folder.glob("*.txt"):

        with open(label_file, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        new_lines = []
        file_changed = False

        for line in lines:

            parts = line.split()

            # Una etiqueta YOLO de detección tiene 5 valores
            if len(parts) == 5:
                new_lines.append(line)
                continue

            # Un polígono necesita:
            # class_id + pares de coordenadas
            if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:

                class_id = parts[0]

                try:
                    coordinates = [float(value) for value in parts[1:]]
                except ValueError:
                    new_lines.append(line)
                    continue

                x_values = coordinates[0::2]
                y_values = coordinates[1::2]

                xmin = min(x_values)
                xmax = max(x_values)
                ymin = min(y_values)
                ymax = max(y_values)

                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2

                width = xmax - xmin
                height = ymax - ymin

                new_line = (
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

                new_lines.append(new_line)
                file_changed = True

            else:
                new_lines.append(line)

        if file_changed:

            # Crear la ruta correspondiente para el backup
            relative_path = label_file.relative_to(DATASET_DIR)
            backup_path = BACKUP_DIR / relative_path

            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # Guardar copia original
            shutil.copy2(label_file, backup_path)

            # Guardar etiqueta corregida
            with open(label_file, "w", encoding="utf-8") as file:
                file.write("\n".join(new_lines) + "\n")

            converted += 1

            print(f"\nConvertido: {relative_path}")

print("\n" + "=" * 50)
print(f"Archivos convertidos: {converted}")
print(f"Copia de seguridad: {BACKUP_DIR}")
print("=" * 50)