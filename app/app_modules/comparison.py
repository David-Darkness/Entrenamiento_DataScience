"""Construcción de tablas comparativas y exportación CSV."""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List

from app_modules.inference import InferenceResult
from app_modules.model_manager import EXPECTED_CLASSES


CLASS_NAMES = [EXPECTED_CLASSES[i] for i in sorted(EXPECTED_CLASSES)]


def build_summary_rows(results: List[InferenceResult]) -> List[Dict[str, Any]]:
    """Genera filas de resumen: un modelo por fila con conteos por clase."""
    rows: List[Dict[str, Any]] = []
    for r in results:
        counts = r.counts_by_class()
        row: Dict[str, Any] = {
            "Modelo": r.model_name,
            "Estado": "OK" if r.success else ("Error" if r.error else "No ejecutado"),
            "Detecciones totales": r.total_detections,
        }
        for cls in CLASS_NAMES:
            row[cls] = counts.get(cls, 0)
        row["Tiempo de inferencia (s)"] = round(r.inference_time, 4)
        row["Dispositivo"] = r.device
        if r.error:
            row["Mensaje de error"] = r.error
        rows.append(row)
    return rows


def build_detection_rows(results: List[InferenceResult]) -> List[Dict[str, Any]]:
    """Genera filas individuales: una detección por fila."""
    rows: List[Dict[str, Any]] = []
    for r in results:
        if not r.success:
            continue
        for det in r.detections:
            row = {
                "Modelo": r.model_name,
                "Estado": "OK",
                "Dispositivo": r.device,
                "Clase": det.class_name,
                "Confianza": round(det.confidence, 4),
                "x1": round(det.x1, 2),
                "y1": round(det.y1, 2),
                "x2": round(det.x2, 2),
                "y2": round(det.y2, 2),
            }
            rows.append(row)
    return rows


def _dicts_to_csv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def export_summary_csv(results: List[InferenceResult]) -> str:
    return _dicts_to_csv(build_summary_rows(results))


def export_detections_csv(results: List[InferenceResult]) -> str:
    return _dicts_to_csv(build_detection_rows(results))
