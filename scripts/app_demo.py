"""
=============================================================================
SMARTROAD AI — PLATAFORMA DEMO DE PRUEBA DE MODELOS
=============================================================================
Aplicación web interactiva desarrollada con Streamlit para la inferencia,
evaluación visual y comparación de modelos de visión por computadora
entrenados en el proyecto SmartRoad AI.

Clases soportadas (6):
- 0: Accidente
- 1: Bus
- 2: Camion
- 3: Carro
- 4: Fire
- 5: Moto
=============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuración de Rutas del Proyecto (Regla AGENTS.md: rutas absolutas)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Rutas estándar de modelos
MODEL_CANDIDATES = {
    "YOLOv8n (Definitivo - 20 Epochs)": [
        PROJECT_DIR / "runs" / "experiments" / "yolov8" / "yolov8n_20epochs-2" / "weights" / "best.pt",
        PROJECT_DIR / "experiments" / "yolov8" / "entrenamiento_02_20_epochs_definitivo" / "weights" / "best.pt",
        PROJECT_DIR / "models" / "base" / "yolov8n.pt",
    ],
    "YOLOv10n (Entrenado - 20 Epochs)": [
        PROJECT_DIR / "experiments" / "yolov10" / "entrenamiento_01_20_epochs" / "weights" / "best.pt",
        PROJECT_DIR / "models" / "base" / "yolov10n.pt",
    ],
    "YOLO26n (Preentrenado Base)": [
        PROJECT_DIR / "models" / "base" / "yolo26n.pt",
        PROJECT_DIR / "yolo26n.pt",
    ],
}

# Clases oficiales de SmartRoad AI y sus colores de visualización
CLASS_METADATA = {
    "Accidente": {"color": "#E63946", "icon": "💥", "is_alert": True, "level": "CRÍTICO"},
    "Fire":      {"color": "#FF5400", "icon": "🔥", "is_alert": True, "level": "CRÍTICO"},
    "Carro":     {"color": "#4361EE", "icon": "🚗", "is_alert": False, "level": "NORMAL"},
    "Bus":       {"color": "#3A0CA3", "icon": "🚌", "is_alert": False, "level": "NORMAL"},
    "Camion":    {"color": "#7209B7", "icon": "🚛", "is_alert": False, "level": "NORMAL"},
    "Moto":      {"color": "#4CC9F0", "icon": "🏍️", "is_alert": False, "level": "NORMAL"},
}

SAMPLE_IMAGES_DIR = PROJECT_DIR / "external_data" / "images"

# ---------------------------------------------------------------------------
# Configuración General de la Página Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartRoad AI | Detección de Incidentes Viales",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Estilos CSS Personalizados para UI Moderna e Intuitiva
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Estilos globales */
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        border: 1px solid #334155;
    }
    .main-header h1 {
        color: #38BDF8;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    
    /* Alertas visuales llamativas */
    .alert-box-critical {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.2rem;
        color: #FCA5A5;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    .alert-box-warning {
        background: rgba(245, 158, 11, 0.15);
        border: 2px solid #F59E0B;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.2rem;
        color: #FCD34D;
        font-weight: 600;
    }
    .alert-box-safe {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.2rem;
        color: #6EE7B7;
        font-weight: 600;
    }
    
    /* Tarjetas de métricas */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
        margin-top: 0.2rem;
    }
    
    /* Badge de clases */
    .badge-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Funciones Utilitarias y Caché de Modelos
# ---------------------------------------------------------------------------
def resolve_model_path(candidate_paths: List[Path]) -> Optional[Path]:
    """Busca la primera ruta válida dentro de los candidatos."""
    for p in candidate_paths:
        if p.exists() and p.is_file():
            return p
    return None


@st.cache_resource(show_spinner="Cargando arquitectura del modelo...")
def load_yolo_model(model_path_str: str) -> YOLO:
    """Carga y cachea el modelo YOLO en memoria."""
    return YOLO(model_path_str)


def detect_device() -> str:
    """Detecta si hay GPU CUDA disponible."""
    return "0" if torch.cuda.is_available() else "cpu"


def run_inference(
    model: YOLO,
    image_np: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    selected_classes: Optional[List[str]] = None,
    device: str = "auto",
) -> Tuple[np.ndarray, List[Dict], float]:
    """
    Ejecuta la inferencia YOLO completamente en memoria (save=False).
    Cumple la regla AGENTS.md de no generar carpetas 'runs/' ni tocar el disco.
    """
    dev = detect_device() if device == "auto" else device

    # Si se filtran clases, obtener los IDs correspondientes
    class_indices = None
    if selected_classes and model.names:
        name_to_idx = {name: idx for idx, name in model.names.items()}
        class_indices = [name_to_idx[name] for name in selected_classes if name in name_to_idx]

    t0 = time.perf_counter()
    results = model.predict(
        source=image_np,
        conf=conf_thresh,
        iou=iou_thresh,
        classes=class_indices,
        device=dev,
        save=False,      # ¡CRUCIAL!: No guarda nada en runs/
        verbose=False,
    )
    inference_time_ms = (time.perf_counter() - t0) * 1000.0

    detections = []
    annotated_img_rgb = image_np.copy()

    if results and len(results) > 0:
        res = results[0]
        # In-memory plot de Ultralytics (retorna formato BGR)
        plotted_bgr = res.plot()
        annotated_img_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"Class {cls_id}")
                conf = float(box.conf[0].item())
                xyxy = [round(float(coord), 1) for coord in box.xyxy[0].tolist()]

                detections.append({
                    "id": i + 1,
                    "clase": cls_name,
                    "confianza": conf,
                    "bbox_xyxy": xyxy,
                })

    return annotated_img_rgb, detections, inference_time_ms


# ---------------------------------------------------------------------------
# Header Principal
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🚦 SmartRoad AI — Plataforma de Pruebas</h1>
        <p>Sistema Inteligente de Detección de Incidentes Viales, Incendios y Vehículos en Tiempo Real</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Barra Lateral (Sidebar) — Parámetros y Configuración
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/traffic-jam.png",
    width=70,
)
st.sidebar.title("⚙️ Control de Inferencia")

# 1. Selección de Modelo
available_model_options = {}
for name, candidates in MODEL_CANDIDATES.items():
    found_path = resolve_model_path(candidates)
    if found_path:
        available_model_options[name] = found_path

# Opción de carga personalizada
available_model_options["Cargar otro archivo .pt..."] = None

selected_model_name = st.sidebar.selectbox(
    "Modelo a evaluar:",
    options=list(available_model_options.keys()),
    index=0,
    help="Elige qué modelo entrenado ejecutar sobre la imagen.",
)

active_model_path: Optional[Path] = None
if selected_model_name == "Cargar otro archivo .pt...":
    custom_path_input = st.sidebar.text_input(
        "Ruta absoluta o relativa del archivo .pt:",
        placeholder="ej: models/base/yolov8n.pt",
    )
    if custom_path_input and Path(custom_path_input).exists():
        active_model_path = Path(custom_path_input)
    else:
        st.sidebar.info("Ingresa una ruta válida de modelo `.pt`.")
else:
    active_model_path = available_model_options[selected_model_name]

if active_model_path:
    st.sidebar.caption(f"📁 **Ruta:** `{active_model_path.relative_to(PROJECT_DIR) if active_model_path.is_relative_to(PROJECT_DIR) else active_model_path}`")
else:
    st.sidebar.error("⚠️ No se encontró el archivo de pesos para este modelo.")

st.sidebar.divider()

# 2. Modo de Operación
mode = st.sidebar.radio(
    "Modo de Prueba:",
    options=["Inferencia Individual", "Comparativa Cara a Cara (YOLOv8 vs YOLOv10)"],
    index=0,
)

st.sidebar.divider()

# 3. Hiperparámetros de Inferencia
st.sidebar.subheader("🎛️ Umbrales de Detección")
conf_thresh = st.sidebar.slider(
    "Umbral de Confianza (Confidence):",
    min_value=0.05,
    max_value=1.00,
    value=0.25,
    step=0.05,
    help="Confianza mínima requerida para considerar una detección válida.",
)

iou_thresh = st.sidebar.slider(
    "Umbral de Supresión No Máxima (IoU / NMS):",
    min_value=0.10,
    max_value=0.95,
    value=0.45,
    step=0.05,
    help="Umbral de solapamiento para filtrar cajas duplicadas.",
)

# 4. Filtro por Clases
all_classes = list(CLASS_METADATA.keys())
selected_classes = st.sidebar.multiselect(
    "Filtrar por Clases Específicas:",
    options=all_classes,
    default=all_classes,
    help="Selecciona qué tipos de objetos quieres detectar (deja todas para análisis completo).",
)

# 5. Hardware Info
device_choice = "0" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
st.sidebar.info(f"⚡ **Aceleración:** {'GPU (' + gpu_name + ')' if device_choice == '0' else 'CPU'}")


# ---------------------------------------------------------------------------
# Selección y Carga de Imagen (Upload vs Galería)
# ---------------------------------------------------------------------------
st.subheader("📸 Entrada de Imagen")

input_tab_upload, input_tab_gallery = st.tabs(["📤 Subir Imagen Propia", "🖼️ Galería de Muestras Rápidas"])

current_image: Optional[Image.Image] = None
current_image_name: str = "imagen_prueba.jpg"

with input_tab_upload:
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona una imagen (JPG, PNG, WEBP):",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploader",
    )
    if uploaded_file is not None:
        try:
            current_image = Image.open(uploaded_file).convert("RGB")
            current_image_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error al abrir la imagen: {e}")

with input_tab_gallery:
    sample_files: List[Path] = []
    if SAMPLE_IMAGES_DIR.exists():
        sample_files = sorted(
            [f for f in SAMPLE_IMAGES_DIR.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]],
            key=lambda x: x.name,
        )

    if sample_files:
        st.write("Haz clic en una imagen de prueba de la galería vial para cargarla instantáneamente:")
        
        # Grid de selección de imágenes
        cols_per_row = 6
        selected_sample_idx = st.session_state.get("selected_sample_idx", None)

        for row_start in range(0, len(sample_files), cols_per_row):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                file_idx = row_start + i
                if file_idx < len(sample_files):
                    f = sample_files[file_idx]
                    with cols[i]:
                        try:
                            thumb = Image.open(f)
                            st.image(thumb, use_container_width=True)
                            btn_label = f.stem[:12] + ("…" if len(f.stem) > 12 else "")
                            if st.button(f"🔍 {btn_label}", key=f"sample_{file_idx}", use_container_width=True):
                                st.session_state["selected_sample_idx"] = file_idx
                                st.session_state["active_sample_path"] = str(f)
                                st.rerun()
                        except Exception:
                            st.caption(f.name)

        if "active_sample_path" in st.session_state and uploaded_file is None:
            active_p = Path(st.session_state["active_sample_path"])
            if active_p.exists():
                current_image = Image.open(active_p).convert("RGB")
                current_image_name = active_p.name
                st.success(f"Muestra activa: **{current_image_name}**")
    else:
        st.info("No se encontraron imágenes en `external_data/images`. Puedes subir tus propias imágenes.")


# ---------------------------------------------------------------------------
# Procesamiento e Inferencia
# ---------------------------------------------------------------------------
if current_image is None:
    st.info("👆 Selecciona una imagen de la galería de muestras o sube una imagen arriba para iniciar el análisis.")
else:
    # Convertir PIL a NumPy RGB
    img_np = np.array(current_image)

    # =======================================================================
    # CASO 1: Inferencia Individual
    # =======================================================================
    if mode == "Inferencia Individual":
        if not active_model_path:
            st.error("Por favor selecciona un modelo con pesos disponibles.")
        else:
            try:
                model = load_yolo_model(str(active_model_path))
                
                # Ejecución de la inferencia
                with st.spinner("Analizando escena vial con SmartRoad AI..."):
                    annotated_rgb, detections, inf_time = run_inference(
                        model=model,
                        image_np=img_np,
                        conf_thresh=conf_thresh,
                        iou_thresh=iou_thresh,
                        selected_classes=selected_classes,
                        device=device_choice,
                    )

                # Detección de incidentes críticos
                has_accident = any(d["clase"] == "Accidente" for d in detections)
                has_fire = any(d["clase"] == "Fire" for d in detections)
                accident_count = sum(1 for d in detections if d["clase"] == "Accidente")
                fire_count = sum(1 for d in detections if d["clase"] == "Fire")

                # Banner de Estado / Alertas
                if has_accident and has_fire:
                    st.markdown(
                        f"""
                        <div class="alert-box-critical">
                            🚨🔥 <strong>EMERGENCIA MÁXIMA EN LA VÍA:</strong> Se detectaron {accident_count} accidente(s) y {fire_count} foco(s) de incendio. Protocolo de auxilio y bomberos requerido.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif has_accident:
                    st.markdown(
                        f"""
                        <div class="alert-box-critical">
                            🚨 <strong>ALERTA DE ACCIDENTE VIAL:</strong> Se ha detectado {accident_count} colisión/accidente en la vía. Precaución y asistencia requerida.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif has_fire:
                    st.markdown(
                        f"""
                        <div class="alert-box-critical">
                            🔥 <strong>ALERTA DE INCENDIO:</strong> Se detectaron {fire_count} punto(s) con fuego en el área. Alerta a unidades de emergencia.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif len(detections) > 0:
                    st.markdown(
                        """
                        <div class="alert-box-safe">
                            🟢 <strong>TRÁFICO REGULAR / SIN INCIDENTES:</strong> Se detecta flujo vehicular normal en la vía.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="alert-box-warning">
                            ℹ️ <strong>SIN DETECCIONES:</strong> No se detectaron objetos con el umbral actual de confianza. Puedes intentar reducir el slider de confianza.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Métricas rápidas (KPIs)
                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                with kpi_col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Objetos Detectados</div>
                            <div class="metric-value">{len(detections)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with kpi_col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Tiempo Inferencia</div>
                            <div class="metric-value">{inf_time:.1f} <span style="font-size:1rem;color:#94A3B8;">ms</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with kpi_col3:
                    fps_est = (1000.0 / inf_time) if inf_time > 0 else 0
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Velocidad Estimada</div>
                            <div class="metric-value">{fps_est:.0f} <span style="font-size:1rem;color:#94A3B8;">FPS</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with kpi_col4:
                    incident_total = accident_count + fire_count
                    color_val = "#EF4444" if incident_total > 0 else "#10B981"
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Incidentes Críticos</div>
                            <div class="metric-value" style="color:{color_val};">{incident_total}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.write("")

                # Comparativa Visual de Imágenes
                col_img_orig, col_img_pred = st.columns(2)
                with col_img_orig:
                    st.markdown("#### 📷 Imagen Original")
                    st.image(current_image, use_container_width=True)

                with col_img_pred:
                    st.markdown(f"#### 🎯 Detecciones ({selected_model_name.split(' ')[0]})")
                    st.image(annotated_rgb, use_container_width=True)

                # Tabla detallada y desglose por clases
                st.divider()
                st.subheader("📊 Desglose de Detecciones")

                col_tab, col_summary = st.columns([2, 1])

                with col_summary:
                    st.markdown("##### Conteo por Categoría")
                    class_counts = {}
                    for d in detections:
                        c = d["clase"]
                        class_counts[c] = class_counts.get(c, 0) + 1

                    for cls_name in all_classes:
                        count = class_counts.get(cls_name, 0)
                        meta = CLASS_METADATA.get(cls_name, {"color": "#64748B", "icon": "📦"})
                        bg_style = f"background-color: {meta['color']};"
                        st.markdown(
                            f"""
                            <div style="display:flex; justify-content:space-between; align-items:center; padding: 6px 12px; margin-bottom: 6px; background:#1E293B; border-radius:8px;">
                                <span><span class="badge-chip" style="{bg_style}">{meta['icon']} {cls_name}</span></span>
                                <strong style="font-size:1.1rem; color:white;">{count}</strong>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with col_tab:
                    st.markdown("##### Telemetría de Cajas Delimitadoras")
                    if detections:
                        table_data = []
                        for d in detections:
                            meta = CLASS_METADATA.get(d["clase"], {"icon": ""})
                            table_data.append({
                                "#": d["id"],
                                "Clase": f"{meta['icon']} {d['clase']}",
                                "Confianza": f"{d['confianza'] * 100:.1f}%",
                                "Caja [x1, y1, x2, y2]": str(d["bbox_xyxy"]),
                            })
                        st.dataframe(table_data, use_container_width=True)
                    else:
                        st.info("Sin registros de cajas delimitadoras con el umbral actual.")

                # Opciones de Descarga
                st.divider()
                st.subheader("💾 Exportar Resultados")
                dl_col1, dl_col2 = st.columns(2)

                # Convertir imagen anotada a bytes para descarga
                annotated_pil = Image.fromarray(annotated_rgb)
                import io
                buf = io.BytesIO()
                annotated_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()

                with dl_col1:
                    st.download_button(
                        label="📥 Descargar Imagen Anotada (.PNG)",
                        data=byte_im,
                        file_name=f"smartroad_pred_{Path(current_image_name).stem}.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                with dl_col2:
                    report_json = {
                        "archivo": current_image_name,
                        "modelo": selected_model_name,
                        "tiempo_inferencia_ms": round(inf_time, 2),
                        "total_detecciones": len(detections),
                        "incidentes_criticos": accident_count + fire_count,
                        "detecciones": detections,
                    }
                    st.download_button(
                        label="📄 Descargar Reporte Telemetría (.JSON)",
                        data=json.dumps(report_json, indent=2, ensure_ascii=False),
                        file_name=f"smartroad_report_{Path(current_image_name).stem}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Error durante la inferencia: {e}")
                st.exception(e)

    # =======================================================================
    # CASO 2: Comparativa Cara a Cara (YOLOv8 vs YOLOv10)
    # =======================================================================
    elif mode == "Comparativa Cara a Cara (YOLOv8 vs YOLOv10)":
        st.subheader("⚔️ Comparativa Directa: YOLOv8n vs YOLOv10n")
        
        path_v8 = resolve_model_path(MODEL_CANDIDATES["YOLOv8n (Definitivo - 20 Epochs)"])
        path_v10 = resolve_model_path(MODEL_CANDIDATES["YOLOv10n (Entrenado - 20 Epochs)"])

        if not path_v8 or not path_v10:
            st.error("Se necesitan ambos modelos (YOLOv8 y YOLOv10) disponibles en el proyecto para la comparativa.")
        else:
            try:
                m_v8 = load_yolo_model(str(path_v8))
                m_v10 = load_yolo_model(str(path_v10))

                with st.spinner("Ejecutando inferencia dual en paralelo..."):
                    img_v8, det_v8, time_v8 = run_inference(
                        model=m_v8,
                        image_np=img_np,
                        conf_thresh=conf_thresh,
                        iou_thresh=iou_thresh,
                        selected_classes=selected_classes,
                        device=device_choice,
                    )
                    img_v10, det_v10, time_v10 = run_inference(
                        model=m_v10,
                        image_np=img_np,
                        conf_thresh=conf_thresh,
                        iou_thresh=iou_thresh,
                        selected_classes=selected_classes,
                        device=device_choice,
                    )

                comp_col1, comp_col2 = st.columns(2)

                # Columna YOLOv8
                with comp_col1:
                    st.markdown("### 🟦 YOLOv8n (Definitivo)")
                    st.image(img_v8, use_container_width=True)
                    st.metric("Total Detecciones", len(det_v8))
                    st.metric("Tiempo de Inferencia", f"{time_v8:.1f} ms")
                    
                    # Conteo de incidentes
                    acc_8 = sum(1 for d in det_v8 if d["clase"] == "Accidente")
                    fire_8 = sum(1 for d in det_v8 if d["clase"] == "Fire")
                    st.caption(f"🚨 Accidentes: **{acc_8}** | 🔥 Incendios: **{fire_8}**")

                # Columna YOLOv10
                with comp_col2:
                    st.markdown("### 🟪 YOLOv10n (Experimental)")
                    st.image(img_v10, use_container_width=True)
                    st.metric("Total Detecciones", len(det_v10), delta=f"{len(det_v10) - len(det_v8)} vs v8")
                    st.metric("Tiempo de Inferencia", f"{time_v10:.1f} ms", delta=f"{time_v10 - time_v8:.1f} ms", delta_color="inverse")
                    
                    # Conteo de incidentes
                    acc_10 = sum(1 for d in det_v10 if d["clase"] == "Accidente")
                    fire_10 = sum(1 for d in det_v10 if d["clase"] == "Fire")
                    st.caption(f"🚨 Accidentes: **{acc_10}** | 🔥 Incendios: **{fire_10}**")

                st.divider()
                st.info("💡 **Análisis Comparativo:** YOLOv8 cuenta con el entrenamiento definitivo consolidado. YOLOv10 ofrece optimización NMS-free end-to-end en su arquitectura.")

            except Exception as e:
                st.error(f"Error en comparativa: {e}")
                st.exception(e)


# ---------------------------------------------------------------------------
# Escalabilidad Futura (Placeholder amigable)
# ---------------------------------------------------------------------------
with st.expander("🚀 Próxima Fase: Escalabilidad a Video & Streaming en Tiempo Real"):
    st.markdown(
        """
        - **Procesamiento de Video (MP4 / AVI):** Inferencia frame por frame con cálculo de FPS en tiempo real.
        - **Cámaras de Tráfico (RTSP / WebRTC):** Conexión directa a fuentes de video en vivo de autopistas.
        - **Seguimiento Multiobjeto (ByteTrack / BoT-SORT):** Conteo de flujo vehicular continuo por carril.
        - **Notificaciones Automáticas:** Envío de webhook / alerta a central de tráfico ante eventos de Accidente o Fuego.
        """
    )
