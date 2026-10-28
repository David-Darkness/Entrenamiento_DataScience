"""SmartRoad AI — Sistema inteligente de monitoreo vial multimodelo.

Aplicacion Streamlit con tres areas:
  1. Dashboard: resumen del estado del sistema y alertas pendientes.
  2. Centro de monitoreo: carga de imagen, analisis automatico y gestion de eventos.
  3. Laboratorio de modelos: comparacion tecnica individual.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
from PIL import Image

from app_modules.alert_manager import (
    EVENT_STATUS_CONFIRMED,
    EVENT_STATUS_DISCARDED,
    EVENT_STATUS_PENDING,
    MonitoredEvent,
    add_event,
    create_event,
    filter_events,
    update_event_status,
)
from app_modules.comparison import (
    CLASS_NAMES,
    build_detection_rows,
    build_summary_rows,
    export_detections_csv,
    export_summary_csv,
)
from app_modules.decision_engine import (
    DEFAULT_MODEL_QUALITY,
    FusionConfig,
    build_decision_result,
    compute_model_weights,
    leader_model,
)
from app_modules.inference import InferenceResult, run_inference
from app_modules.model_manager import (
    EXPECTED_CLASSES,
    ModelLoadResult,
    build_registry,
    get_available_device,
    load_models,
)
from app_modules.notifications import simulate_notification
from app_modules.traffic_analysis import assess_traffic
from app_modules.visualization import (
    annotated_image_to_pil,
    create_class_counts_chart,
    create_inference_time_chart,
    draw_alert_evidence,
)

# ---------------------------------------------------------------------------
# Configuracion de rutas
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# ---------------------------------------------------------------------------
# Configuracion de la pagina
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartRoad AI — Monitoreo Vial Multimodelo",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Estado de sesion
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        "page": "Dashboard",
        "inference_results": None,
        "decision_result": None,
        "original_image": None,
        "image_hash": None,
        "last_params": None,
        "analysis_timestamp": None,
        "analysis_count": 0,
        "events": [],
        "notification_logs": [],
        "analysis_running": False,
        "last_error": None,
        "current_event_ids": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()

# ---------------------------------------------------------------------------
# Carga de modelos (con cache para evitar recargas)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando y calentando los modelos entrenados...")
def get_loaded_models(project_root: Path) -> Dict[str, ModelLoadResult]:
    registry = build_registry(project_root)
    # Carga los pesos y ejecuta una inferencia de calentamiento en el dispositivo
    # preferido para que el primer modelo no reporte tiempos inflados.
    return load_models(registry, device=get_available_device())


models_state = get_loaded_models(PROJECT_ROOT)
preferred_device = get_available_device()

available_models = {
    key: result for key, result in models_state.items() if result.available
}

# ---------------------------------------------------------------------------
# CSS personalizado
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .sr-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f0f5;
        margin-bottom: 0.2rem;
    }
    .sr-subtitle {
        font-size: 1rem;
        color: #a0a0b0;
        margin-bottom: 1.5rem;
    }
    .sr-card {
        background-color: #25252b;
        border: 1px solid #35353d;
        border-radius: 12px;
        padding: 1.1rem;
        height: 100%;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }
    .sr-card-title {
        font-size: 0.85rem;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .sr-card-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f0f5;
        line-height: 1.1;
    }
    .sr-card-sub {
        font-size: 0.8rem;
        color: #888898;
        margin-top: 0.3rem;
    }
    .sr-section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f0f0f5;
        margin: 1.8rem 0 0.8rem 0;
    }
    .sr-section-subtitle {
        font-size: 0.95rem;
        color: #888898;
        margin-bottom: 1rem;
    }
    .sr-alert-high {
        border-left: 4px solid #e74c3c;
        background-color: rgba(231, 76, 60, 0.08);
    }
    .sr-alert-medium {
        border-left: 4px solid #f39c12;
        background-color: rgba(243, 156, 18, 0.08);
    }
    .sr-alert-low {
        border-left: 4px solid #3498db;
        background-color: rgba(52, 152, 219, 0.08);
    }
    .sr-empty {
        color: #888898;
        font-size: 0.95rem;
        padding: 2rem;
        text-align: center;
        border: 1px dashed #45454d;
        border-radius: 10px;
    }
    .sr-model-ok {
        color: #2ecc71;
        font-weight: 600;
    }
    .sr-model-error {
        color: #e74c3c;
        font-weight: 600;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #45454d;
        border-radius: 12px;
        padding: 1rem;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #00bcd4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers de UI
# ---------------------------------------------------------------------------
def _metric_card(title: str, value: str, subtitle: str = "", color: str = "#f0f0f5") -> str:
    return f"""
    <div class="sr-card">
        <div class="sr-card-title">{title}</div>
        <div class="sr-card-value" style="color:{color};">{value}</div>
        {f'<div class="sr-card-sub">{subtitle}</div>' if subtitle else ""}
    </div>
    """


def _section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="sr-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sr-section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def _compute_image_hash(file: Any) -> str:
    return hashlib.md5(file.getvalue()).hexdigest()


def _build_params_dict(
    confidence: float,
    imgsz: int,
    iou_threshold: float,
    traffic_low: int,
    traffic_high: int,
    gamma: float,
    beta: float,
    tau_alert: float,
    tau_count: float,
) -> Dict[str, Any]:
    return {
        "confidence": float(confidence),
        "imgsz": int(imgsz),
        "iou_threshold": float(iou_threshold),
        "traffic_low": int(traffic_low),
        "traffic_high": int(traffic_high),
        "gamma": float(gamma),
        "beta": float(beta),
        "tau_alert": float(tau_alert),
        "tau_count": float(tau_count),
    }


def _fusion_config(params: Dict[str, Any]) -> FusionConfig:
    """Configuracion del modelo de fusion (Capitulo 6 del informe) a partir de los parametros."""
    return FusionConfig(
        iou_threshold=params["iou_threshold"],
        gamma=params["gamma"],
        beta=params["beta"],
        tau_alert=params["tau_alert"],
        tau_count=params["tau_count"],
    )


def _params_changed(current: Dict[str, Any]) -> bool:
    last = st.session_state.last_params
    if last is None:
        return True
    return any(last.get(k) != v for k, v in current.items())


def _run_full_analysis(
    original_image: Image.Image,
    params: Dict[str, Any],
) -> List[InferenceResult]:
    """Ejecuta los cuatro modelos y actualiza el estado de sesion."""
    st.session_state.analysis_running = True
    st.session_state.last_error = None

    # Se pasa la imagen como arreglo BGR (formato de OpenCV/Ultralytics) para no
    # volver a comprimirla en un archivo temporal antes de la inferencia.
    image_bgr = np.ascontiguousarray(np.array(original_image)[..., ::-1])

    all_results: List[InferenceResult] = []
    progress_bar = st.progress(0.0, text="Iniciando analisis multimodelo...")

    model_order = ["yolov8n", "yolov10n", "yolo11n", "rtdetr_l"]
    for idx, key in enumerate(model_order):
        result_state = models_state.get(key)
        if result_state is None or not result_state.available:
            continue

        progress = (idx + 1) / len(model_order)
        progress_bar.progress(
            progress,
            text=f"Ejecutando {result_state.info.display_name} ({idx + 1}/{len(model_order)})...",
        )

        inf_result = run_inference(
            model=result_state.model,
            model_key=result_state.info.key,
            model_name=result_state.info.display_name,
            image_path=image_bgr,
            names=result_state.names or EXPECTED_CLASSES,
            imgsz=params["imgsz"],
            conf=params["confidence"],
            preferred_device=preferred_device,
            allow_cpu_fallback=True,
        )
        all_results.append(inf_result)

    progress_bar.empty()

    # Motor de decisiones: fusion ponderada del Capitulo 6
    decision = build_decision_result(
        all_results,
        traffic_low_threshold=params["traffic_low"],
        traffic_high_threshold=params["traffic_high"],
        iou_threshold=params["iou_threshold"],
        cfg=_fusion_config(params),
    )

    # Generar eventos con imagen de evidencia
    new_events: List[MonitoredEvent] = []
    for candidate in decision.accident_events + decision.fire_events:
        location = candidate.consensus_boxes[0] if candidate.consensus_boxes else None
        evidence_img = draw_alert_evidence(
            original_image,
            [SimpleNamespace(event_type=candidate.event_type, priority=candidate.priority, location_hint=location)],
        )
        event = create_event(
            event_type=candidate.event_type,
            priority=candidate.priority,
            confidence=candidate.confidence,
            supporting_models=candidate.supporting_models,
            evidence_level=candidate.evidence_level,
            explanation=candidate.explanation,
            evidence_image=evidence_img,
            location_hint=location,
        )
        add_event(st.session_state.events, event)
        new_events.append(event)

    # Simular notificaciones para eventos de alta/media prioridad
    for event in new_events:
        if event.priority in ("Alta", "Media"):
            simulate_notification(event, logs=st.session_state.notification_logs)

    # Guardar en estado de sesion
    st.session_state.current_event_ids = [e.event_id for e in new_events]
    st.session_state.inference_results = all_results
    st.session_state.decision_result = decision
    st.session_state.original_image = original_image
    st.session_state.analysis_timestamp = datetime.now()
    st.session_state.analysis_count += 1
    st.session_state.image_hash = params.get("_image_hash")
    st.session_state.last_params = {k: v for k, v in params.items() if not k.startswith("_")}
    st.session_state.analysis_running = False

    return all_results


# ---------------------------------------------------------------------------
# Navegacion
# ---------------------------------------------------------------------------
st.markdown('<div class="sr-title">SmartRoad AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sr-subtitle">Sistema inteligente de monitoreo vial multimodelo</div>',
    unsafe_allow_html=True,
)

pages = ["Dashboard", "Centro de monitoreo", "Laboratorio de modelos"]
selected_page = st.segmented_control(
    "Navegacion",
    pages,
    default=st.session_state.page,
    label_visibility="collapsed",
)
if selected_page and selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

# Sidebar: estado de modelos y ayuda
with st.sidebar:
    _section_header("Modelos disponibles")
    for result in models_state.values():
        status_class = "sr-model-ok" if result.available else "sr-model-error"
        status_text = "Disponible" if result.available else "No disponible"
        st.markdown(
            f"<div style='margin-bottom:8px;'><strong>{result.info.display_name}</strong> "
            f"<span class='{status_class}'>({status_text})</span></div>",
            unsafe_allow_html=True,
        )
        if not result.available:
            st.caption(f"{result.error or 'Clases inconsistentes'}")

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.85rem;color:#888898;'>
        <strong>Ejecutar manualmente</strong><br>
        <code>python -m streamlit run app/app.py</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not available_models:
    st.error("Ninguno de los cuatro checkpoints pudo cargarse. Verifica las rutas y los pesos entrenados.")
    st.stop()

# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------
def render_dashboard() -> None:
    _section_header("Dashboard", "Resumen del estado del sistema y ultimos analisis")

    decision = st.session_state.decision_result
    has_analysis = decision is not None

    # Metricas principales
    pending = len([e for e in st.session_state.events if e.status == EVENT_STATUS_PENDING])
    accidents = len([e for e in st.session_state.events if e.event_type == "POSIBLE ACCIDENTE"])
    fires = len([e for e in st.session_state.events if e.event_type == "POSIBLE INCENDIO"])
    vehicles = decision.total_vehicles if decision else 0
    traffic_level = decision.traffic_level if decision else "—"

    cols = st.columns(5)
    metrics = [
        ("Analisis realizados", str(st.session_state.analysis_count), "en esta sesion"),
        ("Alertas pendientes", str(pending), "requieren revision"),
        ("Posibles accidentes", str(accidents), "total acumulado"),
        ("Posibles incendios", str(fires), "total acumulado"),
        ("Vehiculos estimados", str(vehicles), f"carga {traffic_level}"),
    ]
    for col, (title, value, sub) in zip(cols, metrics):
        color = "#00bcd4" if "Alertas" not in title else ("#e74c3c" if pending > 0 else "#f0f0f5")
        col.markdown(_metric_card(title, value, sub, color), unsafe_allow_html=True)

    # Boton principal
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("🎛️ Abrir centro de monitoreo", type="primary", use_container_width=True):
        st.session_state.page = "Centro de monitoreo"
        st.rerun()

    # Ultimo analisis
    _section_header("Ultimo analisis")
    if not has_analysis:
        st.markdown(
            '<div class="sr-empty">Aun no se ha realizado ningun analisis. '
            'Carga una imagen en el Centro de monitoreo para comenzar.</div>',
            unsafe_allow_html=True,
        )
        return

    ts = st.session_state.analysis_timestamp
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.session_state.original_image is not None:
            st.image(st.session_state.original_image, caption="Imagen analizada", use_container_width=True)
    with c2:
        st.markdown(f"**Fecha y hora:** {ts.strftime('%Y-%m-%d %H:%M:%S') if ts else '—'}")
        st.markdown(f"**Vehiculos estimados:** {vehicles}")
        st.markdown(f"**Carga vehicular visual:** {traffic_level}")
        st.markdown(f"**Alertas activas del ultimo analisis:** {len(decision.accident_events) + len(decision.fire_events)}")

        if decision.accident_events or decision.fire_events:
            for ev in decision.accident_events + decision.fire_events:
                icon = "🚨" if ev.event_type == "POSIBLE ACCIDENTE" else "🔥"
                st.warning(f"{icon} {ev.event_type} — prioridad {ev.priority} (ŝ = {ev.confidence:.2f})")
        else:
            st.success("No se detectaron eventos criticos en el ultimo analisis.")

    # Historial rapido
    _section_header("Historial de eventos recientes")
    recent = sorted(st.session_state.events, key=lambda e: e.timestamp, reverse=True)[:5]
    if not recent:
        st.markdown('<div class="sr-empty">Sin eventos registrados aun.</div>', unsafe_allow_html=True)
    else:
        st.dataframe(
            [e.to_dict() for e in recent],
            use_container_width=True,
            hide_index=True,
            column_order=["Fecha y hora", "Tipo", "Prioridad", "Estado", "Modelos"],
        )


# ---------------------------------------------------------------------------
# CENTRO DE MONITOREO
# ---------------------------------------------------------------------------
def render_monitoring_center() -> None:
    _section_header("Centro de monitoreo", "Carga una imagen para analizarla automaticamente con los cuatro modelos")

    # Controles de analisis
    with st.container():
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            confidence = st.slider(
                "Umbral de confianza",
                0.05,
                1.0,
                0.25,
                0.05,
                help="Solo se conservan detecciones con confianza igual o mayor.",
            )
        with col_ctrl2:
            imgsz = st.selectbox(
                "Tamano de inferencia",
                [320, 480, 640, 800, 1024],
                index=2,
            )
        with col_ctrl3:
            iou_threshold = st.slider(
                "IoU deduplicacion",
                0.1,
                0.9,
                0.5,
                0.05,
                help="Cajas de la misma clase con IoU mayor o igual se agrupan como un objeto.",
            )

    with st.expander("Parametros avanzados del centro de monitoreo"):
        c1, c2 = st.columns(2)
        with c1:
            traffic_low = st.number_input("Umbral carga baja", 0, 50, 3)
        with c2:
            traffic_high = st.number_input("Umbral carga alta", 1, 100, 10)
        if traffic_low >= traffic_high:
            st.warning("El umbral de carga baja debe ser menor que el de carga alta; se ordenaran automaticamente.")

        st.markdown("**Modelo de fusion (Capitulo 6)**")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            fusion_gamma = st.slider(
                "γ — exponente de los pesos", 1.0, 10.0, 2.0, 0.5,
                help="w_k = P_k^γ / Σ P_j^γ, con P_k = mAP@50 de TEST. Valores altos dan mas autoridad al mejor modelo.",
            )
        with f2:
            fusion_beta = st.slider(
                "β — penalizacion sin lider", 1.0, 4.0, 2.0, 0.5,
                help="Si el modelo lider no participa: Φ = (modelos que coinciden / modelos ejecutados)^β.",
            )
        with f3:
            tau_alert = st.slider(
                "τ alerta (Accidente/Fire)", 0.05, 0.95, 0.35, 0.05,
                help="Solo se emite una alerta si ŝ = s̄·Φ es mayor o igual a este valor. Con 0,65 la precision en TEST supera el 95 %.",
            )
        with f4:
            tau_count = st.slider(
                "τ conteo de vehiculos", 0.05, 0.95, 0.25, 0.05,
                help="Solo se cuenta un vehiculo si su confianza fusionada ŝ es mayor o igual a este valor.",
            )

        available_names = [r.info.display_name for r in models_state.values() if r.available]
        preview_weights = compute_model_weights(available_names, DEFAULT_MODEL_QUALITY, fusion_gamma)
        leader = leader_model(preview_weights)
        st.caption(
            "Pesos actuales: "
            + " · ".join(
                f"{name}{' (lider)' if name == leader else ''} w = {w:.3f}"
                for name, w in sorted(preview_weights.items(), key=lambda kv: -kv[1])
            )
        )

    params = _build_params_dict(
        confidence=confidence,
        imgsz=imgsz,
        iou_threshold=iou_threshold,
        traffic_low=traffic_low,
        traffic_high=traffic_high,
        gamma=fusion_gamma,
        beta=fusion_beta,
        tau_alert=tau_alert,
        tau_count=tau_count,
    )

    # Zona de carga
    uploaded_file = st.file_uploader(
        "Seleccionar imagen (JPG, JPEG o PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="monitoring_uploader",
    )

    if uploaded_file is None:
        st.session_state.inference_results = None
        st.session_state.decision_result = None
        st.session_state.original_image = None
        st.session_state.image_hash = None
        st.session_state.last_params = None
        st.markdown(
            '<div class="sr-empty">Arrastra o selecciona una imagen para iniciar el analisis automatico.</div>',
            unsafe_allow_html=True,
        )
        return

    # Procesar imagen cargada
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo abrir la imagen: {exc}")
        return

    current_hash = _compute_image_hash(uploaded_file)
    params["_image_hash"] = current_hash
    needs_analysis = (current_hash != st.session_state.image_hash) or _params_changed(params)

    if needs_analysis:
        _run_full_analysis(original_image, params)
    elif st.session_state.inference_results is None:
        _run_full_analysis(original_image, params)

    # Mostrar resultados
    decision = st.session_state.decision_result
    results = st.session_state.inference_results
    if decision is None or results is None:
        return

    ts = st.session_state.analysis_timestamp

    # Estado general
    _section_header("Estado de la escena")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(_metric_card("Vehiculos estimados", str(decision.total_vehicles), "tras la fusion"), unsafe_allow_html=True)
    with sc2:
        st.markdown(_metric_card("Carga vehicular", decision.traffic_level, "visual estimada"), unsafe_allow_html=True)
    with sc3:
        n_alerts = len(decision.accident_events) + len(decision.fire_events)
        color = "#e74c3c" if n_alerts > 0 else "#f0f0f5"
        st.markdown(_metric_card("Alertas activas", str(n_alerts), "en esta imagen", color), unsafe_allow_html=True)
    with sc4:
        st.markdown(_metric_card("Hora del analisis", ts.strftime("%H:%M:%S") if ts else "—", ""), unsafe_allow_html=True)

    with st.expander("Detalle del modelo de fusion"):
        weight_rows = [
            {
                "Modelo": name + (" (lider)" if name == decision.leader else ""),
                "P_k (mAP@50 TEST)": DEFAULT_MODEL_QUALITY.get(name, float("nan")),
                "Peso w_k": round(w, 4),
            }
            for name, w in sorted(decision.model_weights.items(), key=lambda kv: -kv[1])
        ]
        st.dataframe(weight_rows, use_container_width=True, hide_index=True)
        st.caption(
            f"Grupos descartados por ŝ < τ: {decision.discarded_vehicles} de vehiculos "
            f"(τ = {params['tau_count']:.2f}) y {decision.discarded_alerts} de eventos criticos "
            f"(τ = {params['tau_alert']:.2f})."
        )

    # Imagen de evidencia y desglose
    _section_header("Evidencia visual")
    ev_col1, ev_col2 = st.columns([3, 2])
    with ev_col1:
        if decision.accident_events or decision.fire_events:
            current_ids = set(st.session_state.current_event_ids)
            evidence_events = [
                e
                for e in st.session_state.events
                if e.event_id in current_ids and e.location_hint is not None
            ]
            evidence_img = draw_alert_evidence(original_image, evidence_events)
            st.image(evidence_img, caption="Eventos criticos detectados", use_container_width=True)
        else:
            st.image(original_image, caption="Sin eventos criticos detectados", use_container_width=True)

    with ev_col2:
        _section_header("Vehiculos por clase")
        for cls in ["Carro", "Moto", "Bus", "Camion"]:
            count = decision.vehicle_counts.get(cls, 0)
            st.markdown(f"<div style='font-size:1rem;margin-bottom:6px;'><strong>{cls}:</strong> {count}</div>", unsafe_allow_html=True)

        traffic = assess_traffic(
            decision.vehicle_counts,
            low_threshold=params["traffic_low"],
            high_threshold=params["traffic_high"],
        )
        st.markdown(
            f"<div style='margin-top:12px;padding:10px;border-radius:8px;background-color:{traffic.level_color};color:white;'>"
            f"<strong>{traffic.level}</strong><br><span style='font-size:0.85rem;'>{traffic.message}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(traffic.note)

    # Alertas
    _section_header("Alertas y eventos candidatos")
    if not decision.accident_events and not decision.fire_events:
        st.success("No se detectaron eventos criticos (Accidente/Fire) en esta imagen.")
    else:
        for candidate in decision.accident_events + decision.fire_events:
            icon = "🚨" if candidate.event_type == "POSIBLE ACCIDENTE" else "🔥"
            alert_class = "sr-alert-high" if candidate.priority == "Alta" else ("sr-alert-medium" if candidate.priority == "Media" else "sr-alert-low")
            with st.container():
                st.markdown(
                    f'<div class="sr-card {alert_class}">'
                    f'<div style="font-size:1.1rem;font-weight:600;">{icon} {candidate.event_type}</div>'
                    f'<div style="color:#a0a0b0;font-size:0.9rem;margin:6px 0;">{candidate.explanation}</div>'
                    f'<div style="font-size:0.85rem;">Prioridad: <strong>{candidate.priority}</strong> · '
                    f'Confianza fusionada ŝ: <strong>{candidate.confidence:.2f}</strong> '
                    f'(s̄ = {candidate.mean_score:.2f} · Φ = {candidate.phi:.2f}) · '
                    f'Modelos: <strong>{", ".join(candidate.supporting_models)}</strong></div>'
                    f'<div style="font-size:0.8rem;color:#a0a0b0;margin-top:4px;">{candidate.evidence_level}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Historial de eventos
    _section_header("Historial de eventos de la sesion")
    f1, f2 = st.columns(2)
    with f1:
        filter_type = st.selectbox("Filtrar por tipo", ["Todos", "POSIBLE ACCIDENTE", "POSIBLE INCENDIO"], key="mc_filter_type")
    with f2:
        filter_status = st.selectbox(
            "Filtrar por estado",
            ["Todos", EVENT_STATUS_PENDING, EVENT_STATUS_CONFIRMED, EVENT_STATUS_DISCARDED],
            key="mc_filter_status",
        )

    events = filter_events(
        st.session_state.events,
        event_type=filter_type if filter_type != "Todos" else None,
        status=filter_status if filter_status != "Todos" else None,
    )

    if not events:
        st.markdown('<div class="sr-empty">No hay eventos con los filtros seleccionados.</div>', unsafe_allow_html=True)
    else:
        for event in events:
            alert_class = "sr-alert-high" if event.priority == "Alta" else ("sr-alert-medium" if event.priority == "Media" else "sr-alert-low")
            with st.container():
                st.markdown(
                    f'<div class="sr-card {alert_class}" style="margin-bottom:10px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="font-weight:600;">{event.event_type}</span>'
                    f'<span style="font-size:0.8rem;color:#888898;">{event.event_id} · {event.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</span>'
                    f'</div>'
                    f'<div style="font-size:0.9rem;color:#a0a0b0;margin:6px 0;">{event.explanation}</div>'
                    f'<div style="font-size:0.85rem;">Prioridad: {event.priority} · Confianza: {event.confidence:.2f} · '
                    f'Estado: <strong>{event.status}</strong> · Modelos: {", ".join(event.supporting_models)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if event.evidence_image is not None:
                    st.image(event.evidence_image, width=280)

                ac1, ac2, ac3 = st.columns([1, 1, 4])
                with ac1:
                    if st.button("Confirmar", key=f"confirm_{event.event_id}"):
                        update_event_status(st.session_state.events, event.event_id, EVENT_STATUS_CONFIRMED)
                        st.rerun()
                with ac2:
                    if st.button("Descartar", key=f"discard_{event.event_id}"):
                        update_event_status(st.session_state.events, event.event_id, EVENT_STATUS_DISCARDED)
                        st.rerun()
                with ac3:
                    if st.button("Simular notificacion", key=f"notify_{event.event_id}"):
                        log = simulate_notification(event, logs=st.session_state.notification_logs)
                        st.info(
                            f"[DEMO] Notificacion {log.notification_id} simulada para {log.recipient_type}. "
                            "No se envio ningun mensaje real."
                        )

        st.dataframe(
            [e.to_dict() for e in events],
            use_container_width=True,
            hide_index=True,
        )

    # Notificaciones DEMO
    with st.expander("Registro de notificaciones DEMO"):
        if st.session_state.notification_logs:
            st.dataframe(
                [log.to_dict() for log in st.session_state.notification_logs],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Aun no se han simulado notificaciones.")


# ---------------------------------------------------------------------------
# LABORATORIO DE MODELOS
# ---------------------------------------------------------------------------
def render_lab() -> None:
    _section_header("Laboratorio de modelos", "Comparacion tecnica individual de los cuatro modelos entrenados")

    if st.session_state.inference_results is None:
        st.markdown(
            '<div class="sr-empty">No hay analisis disponible. Ve al Centro de monitoreo, carga una imagen '
            'y los cuatro modelos se ejecutaran automaticamente. Luego regresa aqui para comparar los resultados.</div>',
            unsafe_allow_html=True,
        )
        return

    results: List[InferenceResult] = st.session_state.inference_results

    _section_header("Resultados individuales")
    tabs = st.tabs([r.model_name for r in results])
    for tab, inf_result in zip(tabs, results):
        with tab:
            if not inf_result.success:
                st.error(f"Error en {inf_result.model_name}: {inf_result.error}")
                continue

            col1, col2 = st.columns([3, 2])
            with col1:
                annotated_pil = annotated_image_to_pil(inf_result.annotated_image)
                if annotated_pil is not None:
                    st.image(annotated_pil, caption=f"Detecciones de {inf_result.model_name}", use_container_width=True)
                    buf = BytesIO()
                    annotated_pil.save(buf, format="PNG")
                    st.download_button(
                        label="Descargar imagen anotada",
                        data=buf.getvalue(),
                        file_name=f"{inf_result.model_key}_anotada.png",
                        mime="image/png",
                        key=f"lab_download_img_{inf_result.model_key}",
                    )
                else:
                    st.warning("No se pudo generar la imagen anotada.")

            with col2:
                st.metric("Detecciones totales", inf_result.total_detections)
                st.metric("Tiempo de inferencia", f"{inf_result.inference_time:.3f} s")
                st.metric("Dispositivo usado", inf_result.device.upper())

                counts = inf_result.counts_by_class()
                if counts:
                    st.markdown("**Conteo por clase**")
                    for cls in CLASS_NAMES:
                        st.write(f"- {cls}: {counts.get(cls, 0)}")
                else:
                    st.info("No se detecto ningun objeto por encima del umbral configurado.")

                if inf_result.total_detections > 0:
                    with st.expander("Ver detecciones"):
                        for i, det in enumerate(inf_result.detections, start=1):
                            st.write(
                                f"{i}. **{det.class_name}** — confianza {det.confidence:.2f}  \n"
                                f"   Caja: ({det.x1:.1f}, {det.y1:.1f}) → ({det.x2:.1f}, {det.y2:.1f})"
                            )

    _section_header("Comparacion automatica entre modelos")
    summary_rows = build_summary_rows(results)
    if summary_rows:
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

        col_csv1, col_csv2 = st.columns(2)
        with col_csv1:
            st.download_button(
                label="Descargar resumen (CSV)",
                data=export_summary_csv(results),
                file_name="comparacion_resumen_smartroad.csv",
                mime="text/csv",
                key="lab_download_summary_csv",
            )
        with col_csv2:
            st.download_button(
                label="Descargar detecciones individuales (CSV)",
                data=export_detections_csv(results),
                file_name="detecciones_individuales_smartroad.csv",
                mime="text/csv",
                key="lab_download_detections_csv",
            )

    _section_header("Graficos comparativos")
    gc1, gc2 = st.columns(2)
    with gc1:
        fig_classes = create_class_counts_chart(results)
        if fig_classes:
            st.plotly_chart(fig_classes, use_container_width=True, key="lab_class_chart")
        else:
            st.info("No hay detecciones para graficar.")
    with gc2:
        fig_time = create_inference_time_chart(results)
        if fig_time:
            st.plotly_chart(fig_time, use_container_width=True, key="lab_time_chart")
        else:
            st.info("No hay tiempos de inferencia para graficar.")

    st.markdown(
        """
        ---
        **Nota metodologica:** cada modelo procesa la imagen de forma independiente.
        El numero de detecciones o la confianza maxima de un modelo no equivalen a una mayor precision global.
        """
    )


# ---------------------------------------------------------------------------
# Renderizado principal
# ---------------------------------------------------------------------------
page = st.session_state.page
if page == "Dashboard":
    render_dashboard()
elif page == "Centro de monitoreo":
    render_monitoring_center()
elif page == "Laboratorio de modelos":
    render_lab()
