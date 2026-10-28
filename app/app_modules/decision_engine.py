"""Motor de decisión: interpreta las predicciones de varios modelos.

Combina detecciones de multiples modelos mediante agrupamiento por clase e IoU,
calcula conteos estimados sin contar objetos repetidos y genera eventos
candidatos para clases criticas (Accidente, Fire) con niveles de consenso.

Limitaciones:
- El agrupamiento por IoU asume que cajas superpuestas de la misma clase
  corresponden al mismo objeto real. Si los modelos desplazan o escalan las
  cajas de forma inconsistente, el conteo puede subestimar o sobreestimar.
- Sin seguimiento temporal no se puede distinguir objetos que aparecen en
  posiciones cercanas en imagenes diferentes.
- El consenso espacial indica coincidencia geometrica, no verdad de campo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app_modules.inference import Detection, InferenceResult


VEHICLE_CLASSES = {"Carro", "Moto", "Bus", "Camion"}
ALERT_CLASSES = {"Accidente", "Fire"}


@dataclass
class Cluster:
    """Grupo de detecciones de la misma clase que probablemente son el mismo objeto."""

    class_name: str
    detections: List[Detection] = field(default_factory=list)
    source_models: set = field(default_factory=set)

    def representative_box(self) -> Tuple[float, float, float, float]:
        """Devuelve el promedio de las cajas agrupadas."""
        if not self.detections:
            return (0.0, 0.0, 0.0, 0.0)
        n = len(self.detections)
        x1 = sum(d.x1 for d in self.detections) / n
        y1 = sum(d.y1 for d in self.detections) / n
        x2 = sum(d.x2 for d in self.detections) / n
        y2 = sum(d.y2 for d in self.detections) / n
        return (x1, y1, x2, y2)

    @property
    def max_confidence(self) -> float:
        return max((d.confidence for d in self.detections), default=0.0)

    @property
    def avg_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)


@dataclass
class EventCandidate:
    """Evento detectado a partir del consenso de modelos."""

    event_type: str
    class_name: str
    confidence: float
    supporting_models: List[str]
    consensus_boxes: List[Tuple[float, float, float, float]]
    evidence_level: str
    priority: str
    explanation: str


@dataclass
class DecisionResult:
    """Resultado estructurado del motor de decisiones."""

    vehicle_counts: Dict[str, int]
    total_vehicles: int
    traffic_level: str
    traffic_message: str
    accident_events: List[EventCandidate]
    fire_events: List[EventCandidate]
    all_clusters: Dict[str, List[Cluster]]
    uncertainty_note: str


def _box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Calcula IoU entre dos cajas [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _cluster_detections(
    detections: List[Detection],
    iou_threshold: float = 0.5,
) -> List[Cluster]:
    """Agrupa detecciones de una misma clase usando IoU greedily."""
    clusters: List[Cluster] = []
    # Ordenar por confianza descendente para que el mas confiable sea semilla.
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

    for det in sorted_dets:
        assigned = False
        for cluster in clusters:
            rep = cluster.representative_box()
            if _box_iou((det.x1, det.y1, det.x2, det.y2), rep) >= iou_threshold:
                cluster.detections.append(det)
                assigned = True
                break
        if not assigned:
            cluster = Cluster(class_name=det.class_name, detections=[det])
            clusters.append(cluster)

    return clusters


def consolidate_detections(
    inference_results: List[InferenceResult],
    iou_threshold: float = 0.5,
) -> Dict[str, List[Cluster]]:
    """Reune y deduplica detecciones de todos los modelos por clase."""
    # Mapear cada deteccion al modelo que la genero para conservar la procedencia.
    detection_source: Dict[int, str] = {}
    by_class: Dict[str, List[Detection]] = {}
    for result in inference_results:
        if not result.success:
            continue
        for det in result.detections:
            by_class.setdefault(det.class_name, []).append(det)
            detection_source[id(det)] = result.model_name

    clusters_by_class: Dict[str, List[Cluster]] = {}
    for class_name, dets in by_class.items():
        clusters = _cluster_detections(dets, iou_threshold=iou_threshold)
        for cluster in clusters:
            cluster.source_models = {
                detection_source[id(d)] for d in cluster.detections
            }
        clusters_by_class[class_name] = clusters

    return clusters_by_class


def _build_event_candidate(
    class_name: str,
    event_type: str,
    clusters: List[Cluster],
    single_model_high_conf: float = 0.7,
) -> List[EventCandidate]:
    """Genera eventos candidatos a partir de clusters de una clase critica."""
    events: List[EventCandidate] = []
    for cluster in clusters:
        n_models = len(cluster.source_models)
        max_conf = cluster.max_confidence
        avg_conf = cluster.avg_confidence
        boxes = [cluster.representative_box()]

        if n_models >= 2:
            evidence_level = "Alta: coincidencia de varios modelos"
            priority = "Alta"
            explanation = (
                f"{class_name} detectado por {n_models} modelos con confianza maxima "
                f"{max_conf:.2f}. Existe coincidencia espacial entre las detecciones."
            )
        elif max_conf >= single_model_high_conf:
            evidence_level = "Media: deteccion de un modelo con alta confianza"
            priority = "Media"
            explanation = (
                f"{class_name} detectado por un unico modelo con confianza {max_conf:.2f}. "
                "No hay coincidencia inter-modelo; se requiere verificacion."
            )
        else:
            evidence_level = "Baja: deteccion aislada de baja confianza"
            priority = "Baja"
            explanation = (
                f"{class_name} detectado por un modelo con confianza baja ({max_conf:.2f}). "
                "Se considera candidato pendiente de revision."
            )

        events.append(
            EventCandidate(
                event_type=event_type,
                class_name=class_name,
                confidence=round(avg_conf, 4),
                supporting_models=sorted(cluster.source_models),
                consensus_boxes=boxes,
                evidence_level=evidence_level,
                priority=priority,
                explanation=explanation,
            )
        )
    return events


def build_decision_result(
    inference_results: List[InferenceResult],
    traffic_low_threshold: int = 3,
    traffic_high_threshold: int = 10,
    iou_threshold: float = 0.5,
    single_model_high_conf: float = 0.7,
) -> DecisionResult:
    """Interpreta las predicciones de todos los modelos y produce decisiones."""
    clusters_by_class = consolidate_detections(
        inference_results, iou_threshold=iou_threshold
    )

    # Conteo de vehiculos deduplicado.
    vehicle_counts: Dict[str, int] = {}
    total_vehicles = 0
    for cls in sorted(VEHICLE_CLASSES):
        n = len(clusters_by_class.get(cls, []))
        vehicle_counts[cls] = n
        total_vehicles += n

    # Carga vehicular visual.
    if total_vehicles <= traffic_low_threshold:
        traffic_level = "Baja"
        traffic_message = (
            f"Se estiman {total_vehicles} vehiculos visibles. Carga visual baja."
        )
    elif total_vehicles <= traffic_high_threshold:
        traffic_level = "Media"
        traffic_message = (
            f"Se estiman {total_vehicles} vehiculos visibles. Carga visual media."
        )
    else:
        traffic_level = "Alta"
        traffic_message = (
            f"Se estiman {total_vehicles} vehiculos visibles. Carga visual alta."
        )

    # Eventos criticos.
    accident_events = _build_event_candidate(
        class_name="Accidente",
        event_type="POSIBLE ACCIDENTE",
        clusters=clusters_by_class.get("Accidente", []),
        single_model_high_conf=single_model_high_conf,
    )
    fire_events = _build_event_candidate(
        class_name="Fire",
        event_type="POSIBLE INCENDIO",
        clusters=clusters_by_class.get("Fire", []),
        single_model_high_conf=single_model_high_conf,
    )

    uncertainty_note = (
        "Los conteos se obtienen agrupando detecciones de distintos modelos por IoU. "
        "Una imagen aislada no permite determinar flujo, velocidad ni congestion reales. "
        "El consenso entre modelos es una senal de apoyo, no una garantia de verdad de campo."
    )

    return DecisionResult(
        vehicle_counts=vehicle_counts,
        total_vehicles=total_vehicles,
        traffic_level=traffic_level,
        traffic_message=traffic_message,
        accident_events=accident_events,
        fire_events=fire_events,
        all_clusters=clusters_by_class,
        uncertainty_note=uncertainty_note,
    )


def count_total_raw_detections(inference_results: List[InferenceResult]) -> int:
    """Devuelve el numero total de detecciones crudas sin deduplicar."""
    return sum(len(r.detections) for r in inference_results if r.success)
