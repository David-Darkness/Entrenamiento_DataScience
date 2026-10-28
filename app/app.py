"""SmartRoad AI — Sistema de Deteccion y Comparacion Multimodelo.

Aplicacion Streamlit con dos areas:
  1. Laboratorio de modelos: comparacion tecnica de los cuatro modelos.
  2. Centro de monitoreo: interpretacion de detecciones, indicadores de
     trafico y gestion de alertas.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

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
    page_title="SmartRoad AI — Comparacion Multimodelo",
    page_icon="🚗",
    layout="wide",
)

st.title("SmartRoad AI — Sistema de Deteccion y Comparacion Multimodelo")
st.markdown(
    """
    Esta aplicacion ejecuta automaticamente cuatro modelos entrenados sobre la
    **misma imagen** y ofrece tanto una comparacion tecnica como un centro de
    monitoreo que interpreta las detecciones.

    **Modelos:** YOLOv8n · YOLOv10n · YOLO11n · RT-DETR-L
    """
)

# ---------------------------------------------------------------------------
# Estado de sesion
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        "inference_results": None,
        "decision_result": None,
        "original_image": None,
        "analysis_timestamp": None,
        "events": [],
        "notification_logs": [],
        "last_analysis_signature": None,
        "current_event_ids": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()

# ---------------------------------------------------------------------------
# Carga de modelos (con cache para evitar recargas por cada interaccion)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando y calentando los modelos entrenados...")
def get_loaded_models(project_root: Path) -> Dict[str, ModelLoadResult]:
    registry = build_registry(project_root)
    # Carga los pesos y ejecuta una inferencia de calentamiento en el dispositivo
    # preferido para que el primer modelo no reporte tiempos inflados.
    return load_models(registry, device=get_available_device())


models_state = get_loaded_models(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Panel de disponibilidad de modelos
# ---------------------------------------------------------------------------
with st.expander("Estado de disponibilidad de los modelos", expanded=True):
    cols = st.columns(len(models_state))
    for col, (key, result) in zip(cols, models_state.items()):
        with col:
            if result.available:
                st.success(f"✅ {result.info.display_name}")
                st.caption("Checkpoint cargado correctamente")
            else:
                st.error(f"❌ {result.info.display_name}")
                reason = result.error or "Clases inconsistentes"
                st.caption(f"{reason}")

available_models = {
    key: result for key, result in models_state.items() if result.available
}

if not available_models:
    st.error(
        "Ninguno de los cuatro checkpoints pudo cargarse. Verifica las rutas y los pesos entrenados."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Barra lateral - controles
# ---------------------------------------------------------------------------
st.sidebar.header("Configuracion de analisis")

uploaded_file = st.sidebar.file_uploader(
    "Cargar imagen",
    type=["jpg", "jpeg", "png"],
    help="Formatos admitidos: JPG, JPEG y PNG.",
)

confidence = st.sidebar.slider(
    "Umbral de confianza",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Solo se conservan las detecciones con confianza igual o mayor a este valor.",
)

imgsz = st.sidebar.selectbox(
    "Tamaño de inferencia",
    options=[320, 480, 640, 800, 1024],
    index=2,
    help="Resolucion de entrada para todos los modelos.",
)

st.sidebar.subheader("Centro de monitoreo")
iou_threshold = st.sidebar.slider(
    "IoU para deduplicar detecciones",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Dos cajas de la misma clase con IoU mayor o igual a este valor se consideran el mismo objeto.",
)

traffic_low = st.sidebar.number_input(
    "Umbral carga baja (vehiculos)",
    min_value=0,
    max_value=50,
    value=3,
    help="Valores menores o iguales se consideran carga visual baja.",
)

traffic_high = st.sidebar.number_input(
    "Umbral carga alta (vehiculos)",
    min_value=1,
    max_value=100,
    value=10,
    help="Valores mayores se consideran carga visual alta.",
)

if traffic_low >= traffic_high:
    st.sidebar.warning(
        "El umbral de carga baja debe ser menor que el de carga alta; se ordenaran automaticamente."
    )

st.sidebar.subheader("Modelo de fusion (Cap. 6)")
fusion_gamma = st.sidebar.slider(
    "γ — exponente de los pesos w_k",
    min_value=1.0,
    max_value=10.0,
    value=2.0,
    step=0.5,
    help="w_k = P_k^γ / Σ P_j^γ, con P_k = mAP@50 de TEST. Valores altos dan mas autoridad al mejor modelo.",
)
fusion_beta = st.sidebar.slider(
    "β — penalizacion sin modelo lider",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
    step=0.5,
    help="Si el modelo lider no participa: Φ = (modelos que coinciden / modelos ejecutados)^β.",
)
tau_alert = st.sidebar.slider(
    "τ alerta (Accidente / Fire)",
    min_value=0.05,
    max_value=0.95,
    value=0.35,
    step=0.05,
    help="Solo se emite una alerta si la confianza fusionada ŝ = s̄·Φ es mayor o igual a este valor.",
)
tau_count = st.sidebar.slider(
    "τ conteo de vehiculos",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Solo se cuenta un vehiculo si su confianza fusionada ŝ es mayor o igual a este valor.",
)

_available_names = [
    result.info.display_name for result in models_state.values() if result.available
]
_preview_weights = compute_model_weights(_available_names, DEFAULT_MODEL_QUALITY, fusion_gamma)
with st.sidebar.expander("Pesos actuales de los modelos"):
    _leader = leader_model(_preview_weights)
    for _name, _w in sorted(_preview_weights.items(), key=lambda kv: -kv[1]):
        _tag = " (lider)" if _name == _leader else ""
        st.write(f"- {_name}{_tag}: P = {DEFAULT_MODEL_QUALITY.get(_name, 0):.3f} → w = {_w:.3f}")

fusion_cfg = FusionConfig(
    iou_threshold=float(iou_threshold),
    gamma=float(fusion_gamma),
    beta=float(fusion_beta),
    tau_alert=float(tau_alert),
    tau_count=float(tau_count),
)

preferred_device = get_available_device()
device_label = "CUDA (GPU)" if preferred_device == "0" else "CPU"
st.sidebar.info(f"Dispositivo preferido detectado: **{device_label}**")

st.sidebar.markdown(
    """
    ---
    **Comandos de ejecucion**

    Desde la raiz del proyecto:
    ```powershell
    python -m streamlit run app/app.py
    ```

    O usando el lanzador:
    ```powershell
    .\\app\\run_demo.bat
    ```
    """
)

# ---------------------------------------------------------------------------
# Vista previa de la imagen y boton de analisis
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        uploaded_bytes = uploaded_file.getvalue()
        original_image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo abrir la imagen: {exc}")
        st.stop()
else:
    st.info("Por favor, carga una imagen desde el panel lateral para comenzar el analisis.")
    st.stop()

st.subheader("Vista previa")
st.image(original_image, caption="Imagen original", use_container_width=True)

run_analysis = st.button(
    "🚀 Analizar imagen con los cuatro modelos",
    type="primary",
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Ejecucion secuencial de los modelos
# ---------------------------------------------------------------------------
if run_analysis:
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

        progress = idx / len(model_order)
        progress_bar.progress(
            progress,
            text=f"Procesando con {result_state.info.display_name} ({idx + 1}/{len(model_order)})...",
        )

        inf_result = run_inference(
            model=result_state.model,
            model_key=result_state.info.key,
            model_name=result_state.info.display_name,
            image_path=image_bgr,
            names=result_state.names or EXPECTED_CLASSES,
            imgsz=int(imgsz),
            conf=float(confidence),
            preferred_device=preferred_device,
            allow_cpu_fallback=True,
        )
        all_results.append(inf_result)

    progress_bar.progress(1.0, text="Analisis completado")
    progress_bar.empty()

    # Motor de decisiones (fusion ponderada del Capitulo 6)
    decision = build_decision_result(
        all_results,
        traffic_low_threshold=int(traffic_low),
        traffic_high_threshold=int(traffic_high),
        iou_threshold=float(iou_threshold),
        cfg=fusion_cfg,
    )

    # Firma del analisis: misma imagen + mismos parametros => no duplicar eventos.
    signature = hashlib.sha1(
        uploaded_bytes
        + repr((float(confidence), int(imgsz), fusion_cfg.iou_threshold, fusion_cfg.gamma,
                fusion_cfg.beta, fusion_cfg.tau_alert, fusion_cfg.tau_count)).encode()
    ).hexdigest()
    is_repeated = signature == st.session_state.last_analysis_signature

    if not is_repeated:
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

        # Simular notificaciones para eventos de prioridad alta o media
        for event in new_events:
            if event.priority in ("Alta", "Media"):
                simulate_notification(event, logs=st.session_state.notification_logs)

        st.session_state.current_event_ids = [e.event_id for e in new_events]
        st.session_state.last_analysis_signature = signature

    # Guardar en estado de sesion
    st.session_state.inference_results = all_results
    st.session_state.decision_result = decision
    st.session_state.original_image = original_image
    st.session_state.analysis_timestamp = datetime.now()

    if is_repeated:
        st.info("Esta imagen ya se analizo con los mismos parametros; no se duplicaron eventos en el historial.")
    st.success("Analisis completado. Consulta los resultados en las pestanas.")

# ---------------------------------------------------------------------------
# Pestañas principales
# ---------------------------------------------------------------------------
tab_lab, tab_monitor = st.tabs(
    ["🔬 Laboratorio de modelos", "🎛️ Centro de monitoreo"]
)

# ===========================================================================
# TAB 1: Laboratorio de modelos
# ===========================================================================
with tab_lab:
    if st.session_state.inference_results is None:
        st.info("Presiona 'Analizar imagen con los cuatro modelos' para ver la comparacion tecnica.")
    else:
        results: List[InferenceResult] = st.session_state.inference_results

        st.subheader("Resultados individuales")
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
                        st.image(
                            annotated_pil,
                            caption=f"Detecciones de {inf_result.model_name}",
                            use_container_width=True,
                        )
                        buf = BytesIO()
                        annotated_pil.save(buf, format="PNG")
                        st.download_button(
                            label="⬇️ Descargar imagen anotada",
                            data=buf.getvalue(),
                            file_name=f"{inf_result.model_key}_anotada.png",
                            mime="image/png",
                            key=f"download_img_{inf_result.model_key}",
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

        st.subheader("Comparacion automatica entre modelos")
        summary_rows = build_summary_rows(results)
        if summary_rows:
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)

            col_csv1, col_csv2 = st.columns(2)
            with col_csv1:
                st.download_button(
                    label="⬇️ Descargar resumen (CSV)",
                    data=export_summary_csv(results),
                    file_name="comparacion_resumen_smartroad.csv",
                    mime="text/csv",
                    key="download_summary_csv",
                )
            with col_csv2:
                st.download_button(
                    label="⬇️ Descargar detecciones individuales (CSV)",
                    data=export_detections_csv(results),
                    file_name="detecciones_individuales_smartroad.csv",
                    mime="text/csv",
                    key="download_detections_csv",
                )
        else:
            st.warning("No hay resultados para comparar.")

        st.subheader("Graficos comparativos")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_classes = create_class_counts_chart(results)
            if fig_classes:
                st.plotly_chart(fig_classes, use_container_width=True, key="class_chart")
            else:
                st.info("No hay detecciones para graficar.")
        with chart_col2:
            fig_time = create_inference_time_chart(results)
            if fig_time:
                st.plotly_chart(fig_time, use_container_width=True, key="time_chart")
            else:
                st.info("No hay tiempos de inferencia para graficar.")

        st.markdown(
            """
            ---
            **Nota metodologica:** cada modelo procesa la imagen de forma independiente.
            El numero de detecciones o la confianza maxima de un modelo no equivalen a
            una mayor precision global. Las comparaciones deben interpretarse como
            resultados individuales y no como una verdad absoluta.
            """
        )

# ===========================================================================
# TAB 2: Centro de monitoreo
# ===========================================================================
with tab_monitor:
    if st.session_state.decision_result is None:
        st.info("Presiona 'Analizar imagen con los cuatro modelos' para activar el centro de monitoreo.")
        st.stop()

    decision = st.session_state.decision_result
    base_image = st.session_state.original_image
    timestamp = st.session_state.analysis_timestamp

    st.subheader("Estado general de la escena")
    scene_cols = st.columns(4)
    scene_cols[0].metric("Vehiculos estimados", decision.total_vehicles)
    scene_cols[1].metric("Alertas activas", len(decision.accident_events) + len(decision.fire_events))
    scene_cols[2].metric("Carga vehicular", decision.traffic_level)
    scene_cols[3].metric("Analisis", timestamp.strftime("%H:%M:%S") if timestamp else "-")

    st.caption(decision.uncertainty_note)

    with st.expander("Detalle del modelo de fusion", expanded=False):
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
            f"(τ = {fusion_cfg.tau_count:.2f}) y {decision.discarded_alerts} de eventos criticos "
            f"(τ = {fusion_cfg.tau_alert:.2f})."
        )

    mon_cols = st.columns([3, 2])
    with mon_cols[0]:
        st.subheader("Imagen de evidencia")
        if decision.accident_events or decision.fire_events:
            current_ids = set(st.session_state.current_event_ids)
            evidence_events = [
                e
                for e in st.session_state.events
                if e.event_id in current_ids and e.location_hint is not None
            ]
            evidence_img = draw_alert_evidence(base_image, evidence_events)
            st.image(evidence_img, caption="Eventos criticos detectados", use_container_width=True)
        else:
            st.image(base_image, caption="Sin eventos criticos detectados", use_container_width=True)

    with mon_cols[1]:
        st.subheader("Carga vehicular visual")
        traffic = assess_traffic(
            decision.vehicle_counts,
            low_threshold=int(traffic_low),
            high_threshold=int(traffic_high),
        )
        st.markdown(
            f"<div style='padding:12px;border-radius:8px;background-color:{traffic.level_color};color:white;'>"
            f"<strong>{traffic.level}</strong><br>{traffic.message}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Desglose por clase (deduplicado):**")
        for cls in sorted(traffic.vehicle_counts):
            st.write(f"- {cls}: {traffic.vehicle_counts[cls]}")
        st.caption(traffic.note)

    # Alertas
    st.subheader("Alertas y eventos candidatos")
    all_alert_candidates = decision.accident_events + decision.fire_events
    if not all_alert_candidates:
        st.success("No se detectaron eventos criticos (Accidente/Fire) en esta imagen.")
    else:
        for candidate in all_alert_candidates:
            icon = "🚨" if candidate.event_type == "POSIBLE ACCIDENTE" else "🔥"
            with st.expander(
                f"{icon} {candidate.event_type} — prioridad {candidate.priority} (ŝ = {candidate.confidence:.2f})",
                expanded=True,
            ):
                alert_cols = st.columns([2, 3])
                with alert_cols[0]:
                    st.markdown(f"**Prioridad:** {candidate.priority}")
                    st.markdown(f"**Confianza fusionada ŝ:** {candidate.confidence:.2f}")
                    st.markdown(f"**s̄ = {candidate.mean_score:.2f} · Φ = {candidate.phi:.2f}**")
                    st.markdown(f"**Modelos:** {', '.join(candidate.supporting_models)}")
                    st.markdown(f"**Evidencia:** {candidate.evidence_level}")
                with alert_cols[1]:
                    st.markdown(f"**Explicacion:** {candidate.explanation}")

    # Historial de eventos
    st.subheader("Historial de eventos de la sesion")
    hist_cols = st.columns(2)
    with hist_cols[0]:
        filter_type = st.selectbox(
            "Filtrar por tipo",
            ["Todos", "POSIBLE ACCIDENTE", "POSIBLE INCENDIO"],
            key="filter_type",
        )
    with hist_cols[1]:
        filter_status = st.selectbox(
            "Filtrar por estado",
            ["Todos", EVENT_STATUS_PENDING, EVENT_STATUS_CONFIRMED, EVENT_STATUS_DISCARDED],
            key="filter_status",
        )

    events = st.session_state.events
    filtered = filter_events(
        events,
        event_type=filter_type if filter_type != "Todos" else None,
        status=filter_status if filter_status != "Todos" else None,
    )

    if not filtered:
        st.info("No hay eventos en el historial con los filtros seleccionados.")
    else:
        for event in filtered:
            card_color = {
                EVENT_STATUS_PENDING: "#fff3cd",
                EVENT_STATUS_CONFIRMED: "#d4edda",
                EVENT_STATUS_DISCARDED: "#f8d7da",
            }.get(event.status, "#ffffff")

            with st.container():
                st.markdown(
                    f"<div style='padding:10px;border-radius:6px;background-color:{card_color};margin-bottom:10px;'>"
                    f"<strong>{event.event_type}</strong> | ID: {event.event_id} | "
                    f"{event.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"Prioridad: {event.priority} | Estado: {event.status}<br>"
                    f"Modelos: {', '.join(event.supporting_models)} | Confianza: {event.confidence:.2f}<br>"
                    f"{event.explanation}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if event.evidence_image is not None:
                    st.image(event.evidence_image, width=300)

                action_cols = st.columns(3)
                with action_cols[0]:
                    if st.button("Confirmar", key=f"confirm_{event.event_id}"):
                        update_event_status(events, event.event_id, EVENT_STATUS_CONFIRMED)
                        st.rerun()
                with action_cols[1]:
                    if st.button("Descartar", key=f"discard_{event.event_id}"):
                        update_event_status(events, event.event_id, EVENT_STATUS_DISCARDED)
                        st.rerun()
                with action_cols[2]:
                    if st.button("Simular notificacion", key=f"notify_{event.event_id}"):
                        log = simulate_notification(event, logs=st.session_state.notification_logs)
                        st.info(
                            f"[DEMO] Notificacion {log.notification_id} simulada para "
                            f"{log.recipient_type} por {log.channel}. No se envio ningun mensaje real."
                        )

        # Tabla resumen del historial
        st.markdown("**Tabla de historial**")
        st.dataframe(
            [e.to_dict() for e in filtered],
            use_container_width=True,
            hide_index=True,
        )

    # Notificaciones simuladas
    with st.expander("Registro de notificaciones DEMO"):
        if st.session_state.notification_logs:
            st.dataframe(
                [log.to_dict() for log in st.session_state.notification_logs],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Aun no se han simulado notificaciones.")
        st.caption(
            "Esta seccion es una simulacion. No se envian mensajes reales ni se contactan servicios de emergencia."
        )

    st.info(
        "El historial se mantiene durante la sesion de Streamlit. Se perdera al reiniciar la aplicacion "
        "a menos que se implemente persistencia externa."
    )
