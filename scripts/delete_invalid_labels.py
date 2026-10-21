import argparse
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
LABELS_FOLDERS = (
    DATASET_DIR / "train" / "labels",
    DATASET_DIR / "valid" / "labels",
    DATASET_DIR / "test" / "labels",
)


def get_invalid_reason(line: str) -> str | None:
    parts = line.split()

    if len(parts) == 5:
        return None

    if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
        try:
            [float(value) for value in parts[1:]]
        except ValueError:
            return "coordenadas no numéricas"
        return None

    return "cantidad de valores incorrecta"


def inspect_label_file(label_file: Path) -> tuple[str | None, list[str]]:
    lines = [
        (line_number, line.strip())
        for line_number, line in enumerate(
            label_file.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if line.strip()
    ]

    if not lines:
        return "archivo vacío", []

    malformed_lines = [
        f"línea {line_number}: {reason}"
        for line_number, line in lines
        if (reason := get_invalid_reason(line)) is not None
    ]

    if malformed_lines:
        return "formato incorrecto", malformed_lines

    return None, []


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detecta archivos de etiquetas vacíos o con líneas que no cumplen "
            "los formatos reconocidos por fix_polygon_labels.py."
        )
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="elimina los archivos encontrados; sin esta opción solo los reporta",
    )
    args = parser.parse_args()

    candidates: list[tuple[Path, str, list[str]]] = []

    for labels_folder in LABELS_FOLDERS:
        if not labels_folder.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de etiquetas: {labels_folder}")

        for label_file in sorted(labels_folder.glob("*.txt")):
            reason, malformed_lines = inspect_label_file(label_file)
            if reason is not None:
                candidates.append((label_file, reason, malformed_lines))

    if not candidates:
        print("No se encontraron archivos de etiquetas vacíos o con formato incorrecto.")
        return

    for label_file, reason, malformed_lines in candidates:
        relative_path = label_file.relative_to(PROJECT_DIR)
        print(f"{relative_path} — {reason}")
        for malformed_line in malformed_lines:
            print(f"  {malformed_line}")

    print(f"\nArchivos encontrados: {len(candidates)}")

    if not args.delete:
        print("Modo de revisión: no se eliminó ningún archivo. Usa --delete para eliminarlos.")
        return

    for label_file, _, _ in candidates:
        label_file.unlink()

    print(f"Archivos eliminados: {len(candidates)}")


if __name__ == "__main__":
    main()
