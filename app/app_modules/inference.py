"""Ejecución de inferencia y parsing de resultados por modelo."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class Detection:
    """Una detección individual."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
        }


@dataclass
class InferenceResult:
    """Resultado completo de inferencia para un modelo."""

    model_key: str
    model_name: str
    success: bool = False
    error: Optional[str] = None
    detections: List[Detection] = field(default_factory=list)
    inference_time: float = 0.0
    device: str = "cpu"
    annotated_image: Optional[np.ndarray] = None

    @property
    def total_detections(self) -> int:
        return len(self.detections)

    def counts_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts


def _parse_detections(result: Any, names: Dict[int, str]) -> List[Detection]:
    """Extrae detecciones estructuradas de un objeto Results de Ultralytics."""
    detections: List[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
    clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)

    for conf, cls_id, coords in zip(confs, clss, xyxy):
        cls_id_int = int(round(float(cls_id)))
        detections.append(
            Detection(
                class_id=cls_id_int,
                class_name=names.get(cls_id_int, f"clase_{cls_id_int}"),
                confidence=float(conf),
                x1=float(coords[0]),
                y1=float(coords[1]),
                x2=float(coords[2]),
                y2=float(coords[3]),
            )
        )
    return detections


def _run_once(
    model: Any,
    image_path: str,
    imgsz: int,
    conf: float,
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str], float]:
    """Ejecuta una inferencia y devuelve éxito, resultados, error y tiempo en segundos."""
    start = time.monotonic()
    try:
        results = model(
            image_path,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        elapsed = time.monotonic() - start
        return True, results, None, elapsed
    except torch.cuda.OutOfMemoryError as exc:
        elapsed = time.monotonic() - start
        return False, None, f"CUDA OOM: {exc}", elapsed
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - start
        return False, None, f"{type(exc).__name__}: {exc}", elapsed


def run_inference(
    model: Any,
    model_key: str,
    model_name: str,
    image_path: str,
    names: Dict[int, str],
    imgsz: int = 640,
    conf: float = 0.25,
    preferred_device: str = "0",
    allow_cpu_fallback: bool = True,
) -> InferenceResult:
    """Ejecuta inferencia, gestionando fallback a CPU si CUDA falla por memoria."""
    result = InferenceResult(model_key=model_key, model_name=model_name)

    devices_to_try: List[str] = [preferred_device]
    if allow_cpu_fallback and preferred_device != "cpu":
        devices_to_try.append("cpu")

    raw_results = None
    used_device = preferred_device
    last_error: Optional[str] = None
    inference_elapsed = 0.0

    for device in devices_to_try:
        success, raw_results, error, elapsed = _run_once(
            model=model,
            image_path=image_path,
            imgsz=imgsz,
            conf=conf,
            device=device,
        )
        if success:
            used_device = device
            inference_elapsed = elapsed
            break
        last_error = error
        if device == preferred_device and "OOM" in (error or ""):
            torch.cuda.empty_cache()
    else:
        result.error = last_error or "Inferencia fallida en todos los dispositivos"
        return result

    result.device = used_device
    result.success = True
    result.inference_time = inference_elapsed

    try:
        if raw_results:
            ultralytics_result = raw_results[0]
            result.detections = _parse_detections(ultralytics_result, names)
            try:
                result.annotated_image = ultralytics_result.plot(
                    line_width=2, font_size=0.6, labels=True
                )
            except Exception:  # noqa: BLE001
                result.annotated_image = None
    except Exception as exc:  # noqa: BLE001
        result.success = False
        result.error = f"Error parseando resultados: {type(exc).__name__}: {exc}"
        result.detections = []
        return result

    return result
