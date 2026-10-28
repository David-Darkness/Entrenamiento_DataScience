"""Motor de decisión: fusión ponderada de las predicciones de los cuatro modelos.

Implementa el modelo matemático de acoplamiento del informe técnico
(Capítulo 6, secciones 6.4 y 6.5):

1. Pesos por modelo:      w_k = P_k^gamma / sum_j P_j^gamma
2. Agrupamiento espacial: detecciones de la misma clase con IoU >= Omega_IoU
3. Fusión geométrica:     B_m = sum(w_k * s_t * B_t) / sum(w_k * s_t)
4. Confianza ponderada:   s_bar_m = sum_k(w_k * s_k) / sum_k(w_k)
                          (s_k = confianza máxima del modelo k dentro del grupo)
5. Consenso asimétrico:   s_hat_m = s_bar_m * Phi(C_m), con modelo líder L = argmax_k P_k
       Phi = 1.0                 si L participa y hay >= 2 modelos
       Phi = 0.90                si solo participa L
       Phi = (|M(C_m)| / K)^beta si L no participa
6. Activación:            el grupo cuenta como objeto / evento si s_hat_m >= tau.

Limitaciones:
- El agrupamiento por IoU asume que cajas superpuestas de la misma clase
  corresponden al mismo objeto real.
- Sin seguimiento temporal no se puede relacionar objetos entre imágenes.
- El consenso espacial indica coincidencia geométrica, no verdad de campo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app_modules.inference import Detection, InferenceResult
from app_modules.traffic_analysis import VEHICLE_CLASSES, assess_traffic

ALERT_CLASSES = {"Accidente", "Fire"}

Box = Tuple[float, float, float, float]

# Calidad P_k de cada modelo: mAP@50 sobre el split TEST (informe, sección 5.6),
# leído de evaluation/<modelo>/test_final/BoxPR_curve.png.
DEFAULT_MODEL_QUALITY: Dict[str, float] = {
    "YOLOv8n": 0.825,
    "YOLOv10n": 0.842,
    "YOLO11n": 0.872,
    "RT-DETR-L": 0.859,
}


@dataclass
class FusionConfig:
    """Hiperparámetros del modelo de fusión (Capítulo 6 del informe)."""

    iou_threshold: float = 0.5          # Omega_IoU
    gamma: float = 2.0                  # exponente de los pesos w_k
    beta: float = 2.0                   # penalización cuando el líder no participa
    leader_alone_factor: float = 0.90   # Phi cuando solo detecta el líder
    tau_alert: float = 0.35             # umbral de activación para Accidente/Fire
    tau_count: float = 0.25             # umbral para contar vehículos
    high_priority: float = 0.60         # s_hat >= este valor -> prioridad Alta
    medium_priority: float = 0.45       # s_hat >= este valor -> prioridad Media
    model_quality: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_QUALITY)
    )


# ---------------------------------------------------------------------------
# Paso 1: pesos y modelo líder
# ---------------------------------------------------------------------------
def compute_model_weights(
    model_names: Iterable[str],
    model_quality: Dict[str, float],
    gamma: float = 2.0,
) -> Dict[str, float]:
    """w_k = P_k^gamma / sum_j P_j^gamma sobre los modelos que se ejecutaron."""
    names = list(dict.fromkeys(model_names))
    if not names:
        return {}
    fallback = min(model_quality.values()) if model_quality else 1.0
    powered = {n: max(model_quality.get(n, fallback), 1e-6) ** gamma for n in names}
    total = sum(powered.values())
    return {n: v / total for n, v in powered.items()}


def leader_model(weights: Dict[str, float]) -> Optional[str]:
    """Modelo líder L = argmax_k P_k (equivale a argmax_k w_k)."""
    return max(weights, key=weights.get) if weights else None


def consensus_factor(
    models_in_cluster: Set[str],
    leader: Optional[str],
    n_models: int,
    beta: float = 2.0,
    leader_alone_factor: float = 0.90,
) -> float:
    """Factor de consenso asimétrico Phi(C_m)."""
    if n_models <= 0 or not models_in_cluster:
        return 0.0
    if leader is not None and leader in models_in_cluster:
        return 1.0 if len(models_in_cluster) >= 2 else leader_alone_factor
    return (len(models_in_cluster) / n_models) ** beta


# ---------------------------------------------------------------------------
# Estructuras
# ---------------------------------------------------------------------------
@dataclass
class TaggedDetection:
    """Detección con el modelo que la generó."""

    model_name: str
    det: Detection

    @property
    def box(self) -> Box:
        return (self.det.x1, self.det.y1, self.det.x2, self.det.y2)


@dataclass
class Cluster:
    """Grupo de detecciones de la misma clase que probablemente son el mismo objeto."""

    class_name: str
    members: List[TaggedDetection] = field(default_factory=list)
    fused_box: Box = (0.0, 0.0, 0.0, 0.0)
    mean_score: float = 0.0     # s_bar_m
    phi: float = 0.0            # Phi(C_m)
    fused_score: float = 0.0    # s_hat_m = s_bar_m * Phi

    @property
    def detections(self) -> List[Detection]:
        return [m.det for m in self.members]

    @property
    def source_models(self) -> Set[str]:
        return {m.model_name for m in self.members}

    @property
    def max_confidence(self) -> float:
        return max((m.det.confidence for m in self.members), default=0.0)

    def per_model_max(self) -> Dict[str, float]:
        best: Dict[str, float] = {}
        for m in self.members:
            best[m.model_name] = max(best.get(m.model_name, 0.0), m.det.confidence)
        return best

    def update_fusion(
        self,
        weights: Dict[str, float],
        leader: Optional[str],
        n_models: int,
        cfg: FusionConfig,
    ) -> None:
        """Recalcula caja fusionada, confianza ponderada, Phi y s_hat."""
        num = [0.0, 0.0, 0.0, 0.0]
        den = 0.0
        for m in self.members:
            a = weights.get(m.model_name, 0.0) * m.det.confidence
            for i, v in enumerate(m.box):
                num[i] += a * v
            den += a
        if den > 0:
            self.fused_box = tuple(v / den for v in num)  # type: ignore[assignment]
        elif self.members:
            self.fused_box = self.members[0].box

        best = self.per_model_max()
        w_sum = sum(weights.get(k, 0.0) for k in best)
        self.mean_score = (
            sum(weights.get(k, 0.0) * s for k, s in best.items()) / w_sum if w_sum > 0 else 0.0
        )
        self.phi = consensus_factor(
            set(best), leader, n_models, cfg.beta, cfg.leader_alone_factor
        )
        self.fused_score = self.mean_score * self.phi

    # Compatibilidad con la versión anterior del motor.
    def representative_box(self) -> Box:
        return self.fused_box

    @property
    def avg_confidence(self) -> float:
        return self.mean_score


@dataclass
class EventCandidate:
    """Evento detectado a partir de la fusión de modelos."""

    event_type: str
    class_name: str
    confidence: float                 # s_hat_m (confianza fusionada)
    supporting_models: List[str]
    consensus_boxes: List[Box]
    evidence_level: str
    priority: str
    explanation: str
    mean_score: float = 0.0           # s_bar_m
    phi: float = 0.0


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
    model_weights: Dict[str, float] = field(default_factory=dict)
    leader: Optional[str] = None
    discarded_alerts: int = 0
    discarded_vehicles: int = 0


# ---------------------------------------------------------------------------
# Paso 2-5: agrupamiento y fusión
# ---------------------------------------------------------------------------
def _box_iou(a: Box, b: Box) -> float:
    """Calcula IoU entre dos cajas [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _cluster_class(
    class_name: str,
    tagged: List[TaggedDetection],
    weights: Dict[str, float],
    leader: Optional[str],
    n_models: int,
    cfg: FusionConfig,
) -> List[Cluster]:
    """Agrupamiento voraz estilo WBF: cada detección (por confianza ponderada
    descendente) se une al grupo cuya caja fusionada tenga mayor IoU >= Omega."""
    clusters: List[Cluster] = []
    ordered = sorted(
        tagged,
        key=lambda t: weights.get(t.model_name, 0.0) * t.det.confidence,
        reverse=True,
    )
    for t in ordered:
        best_iou, best_cluster = 0.0, None
        for c in clusters:
            iou = _box_iou(t.box, c.fused_box)
            if iou >= cfg.iou_threshold and iou > best_iou:
                best_iou, best_cluster = iou, c
        if best_cluster is None:
            best_cluster = Cluster(class_name=class_name)
            clusters.append(best_cluster)
        best_cluster.members.append(t)
        best_cluster.update_fusion(weights, leader, n_models, cfg)
    clusters.sort(key=lambda c: c.fused_score, reverse=True)
    return clusters


def fuse_detections(
    per_model: Dict[str, Sequence[Detection]],
    cfg: Optional[FusionConfig] = None,
) -> Tuple[Dict[str, List[Cluster]], Dict[str, float], Optional[str]]:
    """Fusiona detecciones {modelo: [Detection]} -> grupos por clase.

    Devuelve (grupos_por_clase, pesos, modelo_lider). Es la función que
    comparten la aplicación y el script de evaluación sobre TEST.
    """
    cfg = cfg or FusionConfig()
    weights = compute_model_weights(per_model.keys(), cfg.model_quality, cfg.gamma)
    leader = leader_model(weights)
    n_models = len(weights)

    by_class: Dict[str, List[TaggedDetection]] = {}
    for model_name, dets in per_model.items():
        for d in dets:
            by_class.setdefault(d.class_name, []).append(TaggedDetection(model_name, d))

    clusters = {
        cls: _cluster_class(cls, tagged, weights, leader, n_models, cfg)
        for cls, tagged in by_class.items()
    }
    return clusters, weights, leader


def consolidate_detections(
    inference_results: List[InferenceResult],
    iou_threshold: float = 0.5,
    cfg: Optional[FusionConfig] = None,
) -> Dict[str, List[Cluster]]:
    """Compatibilidad: agrupa y fusiona las detecciones de los modelos exitosos."""
    cfg = cfg or FusionConfig(iou_threshold=iou_threshold)
    per_model = {r.model_name: r.detections for r in inference_results if r.success}
    clusters, _, _ = fuse_detections(per_model, cfg)
    return clusters


# ---------------------------------------------------------------------------
# Eventos críticos
# ---------------------------------------------------------------------------
def _priority(score: float, cfg: FusionConfig) -> str:
    if score >= cfg.high_priority:
        return "Alta"
    if score >= cfg.medium_priority:
        return "Media"
    return "Baja"


def _build_events(
    class_name: str,
    event_type: str,
    clusters: List[Cluster],
    leader: Optional[str],
    cfg: FusionConfig,
) -> Tuple[List[EventCandidate], int]:
    events: List[EventCandidate] = []
    discarded = 0
    for c in clusters:
        if c.fused_score < cfg.tau_alert:
            discarded += 1
            continue
        models = sorted(c.source_models)
        priority = _priority(c.fused_score, cfg)
        if leader in c.source_models and len(models) >= 2:
            evidence = f"Consenso alto: {len(models)} modelos coinciden, incluido el líder"
        elif leader in c.source_models:
            evidence = "Consenso medio: solo el modelo líder detecta el evento"
        elif len(models) >= 2:
            evidence = f"Consenso medio: {len(models)} modelos coinciden sin el líder"
        else:
            evidence = "Consenso bajo: detección aislada sin el modelo líder"
        explanation = (
            f"{class_name} detectado por {', '.join(models)}. "
            f"Confianza ponderada s̄ = {c.mean_score:.2f}, factor de consenso Φ = {c.phi:.2f}, "
            f"confianza fusionada ŝ = {c.fused_score:.2f} (umbral τ = {cfg.tau_alert:.2f})."
        )
        events.append(
            EventCandidate(
                event_type=event_type,
                class_name=class_name,
                confidence=round(c.fused_score, 4),
                supporting_models=models,
                consensus_boxes=[c.fused_box],
                evidence_level=evidence,
                priority=priority,
                explanation=explanation,
                mean_score=round(c.mean_score, 4),
                phi=round(c.phi, 4),
            )
        )
    return events, discarded


def build_decision_result(
    inference_results: List[InferenceResult],
    traffic_low_threshold: int = 3,
    traffic_high_threshold: int = 10,
    iou_threshold: float = 0.5,
    single_model_high_conf: float = 0.7,  # conservado por compatibilidad; ya no se usa
    cfg: Optional[FusionConfig] = None,
) -> DecisionResult:
    """Interpreta las predicciones de todos los modelos mediante la fusión ponderada."""
    cfg = cfg or FusionConfig(iou_threshold=iou_threshold)
    per_model = {r.model_name: r.detections for r in inference_results if r.success}
    clusters_by_class, weights, leader = fuse_detections(per_model, cfg)

    # Conteo de vehículos: solo grupos con s_hat >= tau_count.
    vehicle_counts: Dict[str, int] = {}
    discarded_vehicles = 0
    for cls in sorted(VEHICLE_CLASSES):
        groups = clusters_by_class.get(cls, [])
        kept = sum(1 for c in groups if c.fused_score >= cfg.tau_count)
        vehicle_counts[cls] = kept
        discarded_vehicles += len(groups) - kept

    traffic = assess_traffic(vehicle_counts, traffic_low_threshold, traffic_high_threshold)

    accident_events, d1 = _build_events(
        "Accidente", "POSIBLE ACCIDENTE", clusters_by_class.get("Accidente", []), leader, cfg
    )
    fire_events, d2 = _build_events(
        "Fire", "POSIBLE INCENDIO", clusters_by_class.get("Fire", []), leader, cfg
    )

    uncertainty_note = (
        "Los conteos y alertas provienen de la fusión ponderada de los modelos (pesos por "
        "mAP@50 de TEST, consenso Φ guiado por el modelo líder y umbral τ). Una imagen aislada "
        "no permite determinar flujo, velocidad ni congestión reales; el consenso entre modelos "
        "es una señal de apoyo, no una garantía de verdad de campo."
    )

    return DecisionResult(
        vehicle_counts=vehicle_counts,
        total_vehicles=traffic.total_vehicles,
        traffic_level=traffic.level,
        traffic_message=traffic.message,
        accident_events=accident_events,
        fire_events=fire_events,
        all_clusters=clusters_by_class,
        uncertainty_note=uncertainty_note,
        model_weights=weights,
        leader=leader,
        discarded_alerts=d1 + d2,
        discarded_vehicles=discarded_vehicles,
    )


def count_total_raw_detections(inference_results: List[InferenceResult]) -> int:
    """Devuelve el número total de detecciones crudas sin fusionar."""
    return sum(len(r.detections) for r in inference_results if r.success)
