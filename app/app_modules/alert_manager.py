"""Gestion de eventos/alertas del centro de monitoreo."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


EVENT_STATUS_PENDING = "Pendiente de revision"
EVENT_STATUS_CONFIRMED = "Confirmada"
EVENT_STATUS_DISCARDED = "Descartada"


@dataclass
class MonitoredEvent:
    """Evento generado por el motor de decisiones y revisable por el operador."""

    event_id: str
    event_type: str
    priority: str
    confidence: float
    supporting_models: List[str]
    evidence_level: str
    explanation: str
    timestamp: datetime
    status: str = EVENT_STATUS_PENDING
    evidence_image: Optional[Image.Image] = None
    location_hint: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ID": self.event_id,
            "Fecha y hora": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Tipo": self.event_type,
            "Prioridad": self.priority,
            "Confianza": round(self.confidence, 4),
            "Modelos": ", ".join(self.supporting_models),
            "Evidencia": self.evidence_level,
            "Estado": self.status,
            "Explicacion": self.explanation,
        }


def generate_event_id() -> str:
    """Genera un identificador corto y unico para cada evento."""
    return f"EVT-{uuid.uuid4().hex[:8].upper()}"


def create_event(
    event_type: str,
    priority: str,
    confidence: float,
    supporting_models: List[str],
    evidence_level: str,
    explanation: str,
    evidence_image: Optional[Image.Image] = None,
    location_hint: Optional[Tuple[float, float, float, float]] = None,
) -> MonitoredEvent:
    """Crea un evento con identificador y marca de tiempo."""
    return MonitoredEvent(
        event_id=generate_event_id(),
        event_type=event_type,
        priority=priority,
        confidence=confidence,
        supporting_models=supporting_models,
        evidence_level=evidence_level,
        explanation=explanation,
        timestamp=datetime.now(),
        evidence_image=evidence_image,
        location_hint=location_hint,
    )


def add_event(events: List[MonitoredEvent], event: MonitoredEvent) -> None:
    """Agrega un evento al historial si no existe ya con el mismo ID."""
    if not any(e.event_id == event.event_id for e in events):
        events.append(event)


def update_event_status(
    events: List[MonitoredEvent], event_id: str, new_status: str
) -> bool:
    """Actualiza el estado de un evento."""
    for event in events:
        if event.event_id == event_id:
            event.status = new_status
            return True
    return False


def filter_events(
    events: List[MonitoredEvent],
    event_type: Optional[str] = None,
    status: Optional[str] = None,
) -> List[MonitoredEvent]:
    """Filtra eventos por tipo y/o estado."""
    result = events
    if event_type:
        result = [e for e in result if e.event_type == event_type]
    if status:
        result = [e for e in result if e.status == status]
    return result
