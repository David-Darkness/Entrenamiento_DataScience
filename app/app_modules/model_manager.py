"""Gestión de carga y verificación de los checkpoints entrenados."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from ultralytics import RTDETR, YOLO

# Clases esperadas por el proyecto SmartRoad AI
EXPECTED_CLASSES: Dict[int, str] = {
    0: "Accidente",
    1: "Bus",
    2: "Camion",
    3: "Carro",
    4: "Fire",
    5: "Moto",
}


@dataclass(frozen=True)
class ModelInfo:
    """Metadatos de un modelo registrado."""

    key: str
    display_name: str
    family: str
    checkpoint_path: Path
    loader: Callable[[str], Any]


@dataclass
class ModelLoadResult:
    """Resultado del intento de carga de un modelo."""

    info: ModelInfo
    model: Optional[Any] = None
    error: Optional[str] = None
    names: Optional[Dict[int, str]] = None
    classes_ok: bool = False

    @property
    def available(self) -> bool:
        return self.model is not None and self.classes_ok and self.error is None


def build_registry(project_root: Path) -> Dict[str, ModelInfo]:
    """Construye el registro de checkpoints relativos a la raíz del proyecto."""
    return {
        "yolov8n": ModelInfo(
            key="yolov8n",
            display_name="YOLOv8n",
            family="yolov8",
            checkpoint_path=project_root
            / "experiments"
            / "yolov8"
            / "yolov8n_20epochs-2"
            / "weights"
            / "best.pt",
            loader=lambda p: YOLO(p),
        ),
        "yolov10n": ModelInfo(
            key="yolov10n",
            display_name="YOLOv10n",
            family="yolov10",
            checkpoint_path=project_root
            / "experiments"
            / "yolov10"
            / "entrenamiento_01_20_epochs"
            / "weights"
            / "best.pt",
            loader=lambda p: YOLO(p),
        ),
        "yolo11n": ModelInfo(
            key="yolo11n",
            display_name="YOLO11n",
            family="yolov11",
            checkpoint_path=project_root
            / "experiments"
            / "yolov11"
            / "entrenamiento_02_50_epochs_colab"
            / "weights"
            / "best.pt",
            loader=lambda p: YOLO(p),
        ),
        "rtdetr_l": ModelInfo(
            key="rtdetr_l",
            display_name="RT-DETR-L",
            family="rtdetr",
            checkpoint_path=project_root
            / "experiments"
            / "rtdetr"
            / "entrenamiento_01_25_epochs"
            / "weights"
            / "best.pt",
            loader=lambda p: RTDETR(p),
        ),
    }


def _verify_class_names(names: Dict[int, str]) -> tuple[bool, Optional[str]]:
    """Comprueba que los nombres de clase coincidan con los esperados."""
    if not isinstance(names, dict):
        return False, f"El formato de nombres no es un diccionario: {type(names)}"

    missing = []
    mismatched = []
    for idx, expected in EXPECTED_CLASSES.items():
        actual = names.get(idx)
        if actual is None:
            missing.append(idx)
        elif actual != expected:
            mismatched.append(f"{idx}: esperado '{expected}', obtenido '{actual}'")

    if missing or mismatched:
        parts = []
        if missing:
            parts.append(f"índices faltantes {missing}")
        if mismatched:
            parts.append("; ".join(mismatched))
        return False, "Clases inconsistentes: " + "; ".join(parts)

    return True, None


def load_models(
    registry: Dict[str, ModelInfo],
    device: str | torch.device = "cpu",
) -> Dict[str, ModelLoadResult]:
    """Carga todos los modelos del registro y reporta el estado individual."""
    results: Dict[str, ModelLoadResult] = {}

    for key, info in registry.items():
        result = ModelLoadResult(info=info)

        if not info.checkpoint_path.exists():
            result.error = f"No se encontró el checkpoint: {info.checkpoint_path}"
            results[key] = result
            continue

        try:
            model = info.loader(str(info.checkpoint_path))
            result.model = model
            result.names = dict(model.names) if hasattr(model, "names") else None
            result.classes_ok, class_error = _verify_class_names(result.names or {})
            if not result.classes_ok:
                result.error = class_error
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"

        results[key] = result

    return results


def get_available_device() -> str:
    """Devuelve '0' si hay CUDA disponible, de lo contrario 'cpu'."""
    return "0" if torch.cuda.is_available() else "cpu"
