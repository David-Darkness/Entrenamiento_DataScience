"""Analisis visual de carga vehicular a partir de conteos deduplicados.

Advertencia: una imagen estatica solo permite estimar cuantos vehiculos estan
visibles. No permite calcular velocidad, flujo vehicular por unidad de tiempo
ni congestion real. Estos umbrales son parametros iniciales de demostracion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


VEHICLE_CLASSES = {"Carro", "Moto", "Bus", "Camion"}


@dataclass
class TrafficAssessment:
    """Resultado del analisis de carga vehicular visual."""

    total_vehicles: int
    vehicle_counts: Dict[str, int]
    level: str
    level_color: str
    message: str
    low_threshold: int
    high_threshold: int
    note: str


def assess_traffic(
    vehicle_counts: Dict[str, int],
    low_threshold: int = 3,
    high_threshold: int = 10,
) -> TrafficAssessment:
    """Clasifica la carga vehicular visual segun umbrales configurables."""
    total = sum(vehicle_counts.get(cls, 0) for cls in VEHICLE_CLASSES)

    if total <= low_threshold:
        level = "Baja"
        level_color = "#2ecc71"
        message = f"Carga vehicular visual BAJA ({total} vehiculos estimados)."
    elif total <= high_threshold:
        level = "Media"
        level_color = "#f1c40f"
        message = f"Carga vehicular visual MEDIA ({total} vehiculos estimados)."
    else:
        level = "Alta"
        level_color = "#e74c3c"
        message = f"Carga vehicular visual ALTA ({total} vehiculos estimados)."

    note = (
        "Esta estimacion se basa unicamente en una imagen estatica. "
        "Para determinar velocidad, flujo o congestion reales se requieren "
        "secuencias de video, seguimiento de objetos y una region vial definida."
    )

    return TrafficAssessment(
        total_vehicles=total,
        vehicle_counts=vehicle_counts,
        level=level,
        level_color=level_color,
        message=message,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        note=note,
    )
