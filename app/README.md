# SmartRoad AI — Sistema de Deteccion y Comparacion Multimodelo

Aplicacion web local construida con **Streamlit** que ejecuta los cuatro modelos entrenados del proyecto SmartRoad AI sobre una misma imagen. Ofrece dos areas diferenciadas:

1. **Laboratorio de modelos**: comparacion tecnica de las detecciones individuales.
2. **Centro de monitoreo**: interpretacion de las predicciones, indicadores de trafico y gestion de alertas.

## Modelos soportados

| Modelo | Familia | Checkpoint esperado |
|--------|---------|---------------------|
| YOLOv8n | yolov8 | `experiments/yolov8/yolov8n_20epochs-2/weights/best.pt` |
| YOLOv10n | yolov10 | `experiments/yolov10/entrenamiento_01_20_epochs/weights/best.pt` |
| YOLO11n | yolov11 | `experiments/yolov11/entrenamiento_02_50_epochs_colab/weights/best.pt` |
| RT-DETR-L | rtdetr | `experiments/rtdetr/entrenamiento_01_25_epochs/weights/best.pt` |

Las clases detectadas son: `Accidente`, `Bus`, `Camion`, `Carro`, `Fire`, `Moto`.

## Requisitos previos

- Python 3.10 o superior.
- El entorno de entrenamiento debe tener instalados **PyTorch**, **torchvision** y **Ultralytics** con la version CUDA correspondiente.
- Las dependencias especificas de la aplicacion se instalan desde este directorio.

## Instalacion de dependencias

Activa el entorno de entrenamiento (por ejemplo, `entrenamiento-proyecto`) y ejecuta:

```powershell
python -m pip install -r app/requirements.txt
```

> **Nota:** `requirements.txt` no reinstala PyTorch, CUDA ni Ultralytics. Si faltan, instalalos manualmente siguiendo las instrucciones oficiales de Ultralytics para tu version de CUDA.

## Ejecucion

### Desde la raiz del proyecto (recomendado)

```powershell
python -m streamlit run app/app.py
```

### Usando el lanzador de Windows

```powershell
.\app\run_demo.bat
```

El lanzador detectara el Python del entorno activo, verificara las dependencias criticas e iniciara Streamlit en `http://localhost:8501`.

## Estructura de la aplicacion

```text
app/
├── app.py                       # Interfaz Streamlit y orquestacion
├── requirements.txt             # Dependencias especificas de la app
├── run_demo.bat                # Lanzador para Windows
├── README.md                   # Este archivo
└── app_modules/
    ├── __init__.py
    ├── model_manager.py        # Carga y verificacion de checkpoints
    ├── inference.py            # Inferencia y parsing de resultados
    ├── comparison.py           # Tablas comparativas y exportacion CSV
    ├── visualization.py        # Imagenes anotadas y graficos
    ├── decision_engine.py      # Deduplicacion, consenso y decisiones
    ├── traffic_analysis.py     # Indicador de carga vehicular visual
    ├── alert_manager.py        # Gestion de eventos y estados
    └── notifications.py        # Simulacion DEMO de notificaciones
```

## Funcionalidades

### Laboratorio de modelos

- Carga de imagenes JPG, JPEG y PNG.
- Ejecucion automatica de los cuatro modelos sobre la misma imagen.
- Vista previa de la imagen original e imagenes anotadas por modelo.
- Umbral de confianza configurable (predeterminado `0.25`).
- Tamano de inferencia configurable (predeterminado `640`).
- Tabla comparativa con detecciones totales, conteo por clase y tiempos.
- Graficos comparativos de detecciones por clase y tiempos de inferencia.
- Exportacion de CSV:
  - `comparacion_resumen_smartroad.csv` — resumen por modelo.
  - `detecciones_individuales_smartroad.csv` — detecciones individuales.
- Gestion de errores individual: si un modelo falla, los demas continuan.
- Fallback a CPU si CUDA produce un error de memoria.

### Centro de monitoreo

- **Imagen de evidencia**: imagen original con recuadros sobre eventos criticos.
- **Estado general de la escena**: vehiculos estimados, alertas activas, carga vehicular y hora del analisis.
- **Carga vehicular visual**: total estimado y desglose por clase, con umbrales configurables.
- **Alertas de posibles accidentes**: eventos `POSIBLE ACCIDENTE` con prioridad, modelos que respaldan la alerta y evidencia de consenso.
- **Alertas de posible incendio**: eventos `POSIBLE INCENDIO` con recomendacion de verificacion; no contacta automaticamente a servicios de emergencia.
- **Gestion de estados**: cada evento puede marcarse como `Pendiente de revision`, `Confirmada` o `Descartada`.
- **Historial de eventos**: filtrable por tipo y estado; se mantiene durante la sesion de Streamlit.
- **Notificaciones DEMO**: simula el envio de alertas sin enviar mensajes reales; prepara la arquitectura para canales futuros.

### Reglas del motor de decision

- **Deduplicacion**: las detecciones de la misma clase provenientes de distintos modelos se agrupan mediante IoU. Cada grupo se cuenta como un unico objeto estimado.
- **Consenso para Accidente/Fire**:
  - **Alta**: el evento es detectado por dos o mas modelos con coincidencia espacial.
  - **Media**: un solo modelo detecta el evento con confianza alta (configurable, predeterminado `>= 0.70`).
  - **Baja**: deteccion aislada de baja confianza; se genera una alerta pendiente de verificacion.
- **Carga vehicular visual**: se calcula a partir de los vehiculos deduplicados (`Carro`, `Moto`, `Bus`, `Camion`) y se clasifica con umbrales configurables (baja/medio/alta).

## Limitaciones

- El conteo por IoU asume que cajas superpuestas de la misma clase corresponden al mismo objeto. Si los modelos alinean mal las cajas, el conteo puede ser inexacto.
- Una imagen aislada no permite determinar velocidad, flujo vehicular por unidad de tiempo ni congestion real.
- No se implementa seguimiento de objetos ni procesamiento de video; la arquitectura esta preparada para incorporarlos en una fase posterior.
- No se presenta fusion multimodelo (WBF) como funcionalidad implementada.
- El centro de monitoreo no envia notificaciones reales ni contacta servicios de emergencia; la seccion de notificaciones es una simulacion DEMO.
- El historial de eventos se pierde al reiniciar la sesion de Streamlit, salvo que se implemente persistencia externa.

## Notas importantes

- Los checkpoints entrenados, el dataset y los resultados historicos **no deben moverse ni modificarse**.
- Los resultados mostrados son inferencias individuales por modelo; el numero de detecciones o la confianza maxima de un modelo no equivalen a una mayor precision global.
- El consenso entre modelos es una senal de apoyo, no una garantia de que el evento sea real.
- No se afirma que el sistema mida congestion real ni que envie alertas a servicios de emergencia.
