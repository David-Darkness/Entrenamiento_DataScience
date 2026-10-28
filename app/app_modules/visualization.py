"""Visualización de imágenes anotadas y gráficos comparativos."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

from PIL import ImageDraw, ImageFont

from app_modules.alert_manager import MonitoredEvent
from app_modules.inference import InferenceResult
from app_modules.model_manager import EXPECTED_CLASSES


CLASS_NAMES = [EXPECTED_CLASSES[i] for i in sorted(EXPECTED_CLASSES)]


def annotated_image_to_pil(image: Optional[np.ndarray]) -> Optional[Image.Image]:
    """Convierte una imagen anotada de numpy (BGR) a PIL RGB."""
    if image is None:
        return None
    try:
        # Ultralytics devuelve imágenes en BGR por defecto.
        rgb = image[..., ::-1] if image.shape[-1] == 3 else image
        return Image.fromarray(rgb)
    except Exception:  # noqa: BLE001
        return None


def create_class_counts_chart(results: List[InferenceResult]) -> Optional[Any]:
    """Gráfico de barras agrupadas: detecciones por clase y modelo."""
    records: List[Dict[str, Any]] = []
    for r in results:
        if not r.success:
            continue
        counts = r.counts_by_class()
        for cls in CLASS_NAMES:
            records.append(
                {
                    "Modelo": r.model_name,
                    "Clase": cls,
                    "Detecciones": counts.get(cls, 0),
                }
            )
    if not records:
        return None

    df = pd.DataFrame(records)
    fig = px.bar(
        df,
        x="Clase",
        y="Detecciones",
        color="Modelo",
        barmode="group",
        title="Detecciones por clase y modelo",
        labels={"Detecciones": "Número de detecciones"},
    )
    fig.update_layout(
        xaxis_title="Clase",
        yaxis_title="Detecciones",
        legend_title_text="Modelo",
        template="plotly_white",
    )
    return fig


def create_inference_time_chart(results: List[InferenceResult]) -> Optional[Any]:
    """Gráfico de barras: tiempo de inferencia por modelo."""
    records = [
        {
            "Modelo": r.model_name,
            "Tiempo (s)": round(r.inference_time, 4),
            "Estado": "OK" if r.success else "Error",
        }
        for r in results
    ]
    if not records:
        return None

    df = pd.DataFrame(records)
    fig = px.bar(
        df,
        x="Modelo",
        y="Tiempo (s)",
        color="Estado",
        title="Tiempo de inferencia por modelo",
        labels={"Tiempo (s)": "Segundos"},
        color_discrete_map={"OK": "#2ecc71", "Error": "#e74c3c"},
    )
    fig.update_layout(
        xaxis_title="Modelo",
        yaxis_title="Tiempo (s)",
        template="plotly_white",
    )
    return fig


def draw_alert_evidence(
    base_image: Image.Image,
    events: List[MonitoredEvent],
    color_map: Optional[Dict[str, str]] = None,
) -> Image.Image:
    """Dibuja en la imagen base las ubicaciones de eventos criticos."""
    color_map = color_map or {
        "POSIBLE ACCIDENTE": "#e74c3c",
        "POSIBLE INCENDIO": "#e67e22",
    }
    img = base_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()

    for event in events:
        location = getattr(event, "location_hint", None)
        if location is None:
            continue
        event_type = getattr(event, "event_type", "EVENTO")
        priority = getattr(event, "priority", "")
        color = color_map.get(event_type, "#e74c3c")
        x1, y1, x2, y2 = location
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f"{event_type} ({priority})" if priority else event_type
        draw.text((x1 + 4, max(0, y1 - 18)), label, fill=color, font=font)

    return img
