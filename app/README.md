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
- Calentamiento de los modelos al cargarlos, para que el tiempo de inferencia del primer modelo no incluya su inicializacion.
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
- **Detalle del modelo de fusion**: pesos `w_k` de cada modelo, modelo lider y numero de grupos descartados por el umbral `τ`.
- **Gestion de estados**: cada evento puede marcarse como `Pendiente de revision`, `Confirmada` o `Descartada`.
- **Historial de eventos**: filtrable por tipo y estado; se mantiene durante la sesion de Streamlit. Analizar dos veces la misma imagen con los mismos parametros no duplica eventos.
- **Notificaciones DEMO**: simula el envio de alertas sin enviar mensajes reales; prepara la arquitectura para canales futuros.

### Motor de decision: fusion ponderada (Capitulo 6 del informe tecnico)

`app_modules/decision_engine.py` implementa el modelo matematico de acoplamiento:

1. **Pesos por modelo**: `w_k = P_k^γ / Σ P_j^γ`, con `P_k` = mAP@50 de TEST (`DEFAULT_MODEL_QUALITY`) y `γ = 2` por defecto.
2. **Agrupamiento espacial**: por clase, cada deteccion se une al grupo cuya caja fusionada tenga mayor IoU `>= Ω_IoU` (0.5 por defecto).
3. **Fusion geometrica**: `B = Σ w_k·s·B / Σ w_k·s` (Weighted Boxes Fusion).
4. **Confianza ponderada**: `s̄ = Σ w_k·s_k / Σ w_k`, con `s_k` la confianza maxima del modelo `k` en el grupo.
5. **Consenso asimetrico**: `ŝ = s̄·Φ`, donde el modelo lider es `L = argmax P_k` (YOLO11n) y
   `Φ = 1` si `L` participa con al menos otro modelo, `Φ = 0.90` si solo detecta `L` y `Φ = (n/K)^β` si `L` no participa.
6. **Activacion**: solo se emite una alerta de `Accidente`/`Fire` si `ŝ >= τ_alerta` (0.35) y solo se cuenta un vehiculo si `ŝ >= τ_conteo` (0.25).
   Prioridad: Alta si `ŝ >= 0.60`, Media si `ŝ >= 0.45`, Baja en otro caso.

`γ`, `β`, `τ_alerta`, `τ_conteo` e `Ω_IoU` se ajustan desde la barra lateral. La misma funcion `fuse_detections` se usa en `scripts/evaluate_fusion_test.py` para medir la fusion sobre TEST.

## Limitaciones

- El conteo por IoU asume que cajas superpuestas de la misma clase corresponden al mismo objeto. Si los modelos alinean mal las cajas, el conteo puede ser inexacto.
- Una imagen aislada no permite determinar velocidad, flujo vehicular por unidad de tiempo ni congestion real.
- No se implementa seguimiento de objetos ni procesamiento de video; la arquitectura esta preparada para incorporarlos en una fase posterior.
- La agrupacion es por clase: un mismo vehiculo clasificado como `Carro` por un modelo y `Camion` por otro produce dos grupos.
- El centro de monitoreo no envia notificaciones reales ni contacta servicios de emergencia; la seccion de notificaciones es una simulacion DEMO.
- El historial de eventos se pierde al reiniciar la sesion de Streamlit, salvo que se implemente persistencia externa.

## Notas importantes

- Los checkpoints entrenados, el dataset y los resultados historicos **no deben moverse ni modificarse**.
- Los resultados mostrados son inferencias individuales por modelo; el numero de detecciones o la confianza maxima de un modelo no equivalen a una mayor precision global.
- El consenso entre modelos es una senal de apoyo, no una garantia de que el evento sea real.
- No se afirma que el sistema mida congestion real ni que envie alertas a servicios de emergencia.
