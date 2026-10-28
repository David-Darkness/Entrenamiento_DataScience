# SmartRoad AI — Sistema inteligente de monitoreo vial multimodelo

Aplicacion web local construida con **Streamlit** que ejecuta los cuatro modelos entrenados del proyecto SmartRoad AI sobre una misma imagen. Ofrece tres areas diferenciadas:

1. **Dashboard**: resumen del estado del sistema, modelos disponibles y alertas pendientes.
2. **Centro de monitoreo**: carga de imagen con **analisis automatico**, indicadores de trafico y gestion de alertas.
3. **Laboratorio de modelos**: comparacion tecnica individual de YOLOv8n, YOLOv10n, YOLO11n y RT-DETR-L.

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
├── .streamlit/
│   └── config.toml            # Tema oscuro y configuracion de Streamlit
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

### Dashboard

- Estado de disponibilidad de los cuatro modelos.
- Contadores de analisis realizados, alertas pendientes, posibles accidentes e incendios.
- Resumen del ultimo analisis con imagen y hora.
- Boton principal para abrir el Centro de monitoreo.

### Centro de monitoreo

- **Analisis automatico**: al cargar una imagen se ejecutan los cuatro modelos sin necesidad de pulsar un segundo boton.
- Barra de progreso con el modelo que se esta ejecutando.
- Calentamiento de los modelos al cargarlos, para que el tiempo de inferencia del primer modelo no incluya su inicializacion.
- Imagen de evidencia con recuadros sobre eventos criticos.
- Estado general de la escena: vehiculos estimados, alertas activas y carga vehicular.
- Indicador de **carga vehicular visual** con desglose por clase.
- Alertas de posibles accidentes e incendios con confianza fusionada `ŝ = s̄·Φ`, prioridad, modelos que respaldan la alerta y nivel de consenso.
- Detalle del modelo de fusion: pesos `w_k` de cada modelo, modelo lider y grupos descartados por el umbral `τ`.
- Confirmacion o descarte de eventos por parte del operador.
- Historial de eventos filtrable por tipo y estado.
- Registro de notificaciones DEMO (sin envio real).

### Laboratorio de modelos

- Comparacion tecnica de los cuatro modelos sobre la ultima imagen analizada.
- Imagen anotada por modelo, conteo por clase, coordenadas y confianza.
- Tabla comparativa con tiempos de inferencia.
- Graficos de detecciones por clase y tiempos.
- Exportacion CSV de resumen y detecciones individuales.

## Analisis automatico y control de re-ejecucion

La aplicacion detecta una nueva imagen mediante el **hash MD5** de su contenido. Solo vuelve a ejecutar los cuatro modelos cuando ocurre alguna de estas condiciones:

- Se carga una imagen con hash diferente al ultimo analisis.
- Se modifica algun parametro que afecta el analisis (umbral de confianza, tamano de inferencia, IoU de agrupamiento, umbrales de carga vehicular o parametros de fusion `γ`, `β`, `τ` de alerta y `τ` de conteo).

Si se navega entre paginas, se cambia un widget o se produce un rerun de Streamlit sin cambiar la imagen ni los parametros, **no se re-ejecutan los modelos**.

Al eliminar la imagen del centro de monitoreo se limpian los resultados visuales actuales, pero se conserva el historial de eventos de la sesion.

## Motor de decision: fusion ponderada (Capitulo 6 del informe tecnico)

`app_modules/decision_engine.py` implementa el modelo matematico de acoplamiento:

1. **Pesos por modelo**: `w_k = P_k^γ / Σ P_j^γ`, con `P_k` = mAP@50 de TEST (`DEFAULT_MODEL_QUALITY`) y `γ = 2` por defecto.
2. **Agrupamiento espacial**: por clase, cada deteccion se une al grupo cuya caja fusionada tenga mayor IoU `>= Ω_IoU` (0.5 por defecto).
3. **Fusion geometrica**: `B = Σ w_k·s·B / Σ w_k·s` (Weighted Boxes Fusion).
4. **Confianza ponderada**: `s̄ = Σ w_k·s_k / Σ w_k`, con `s_k` la confianza maxima del modelo `k` en el grupo.
5. **Consenso asimetrico**: `ŝ = s̄·Φ`, donde el modelo lider es `L = argmax P_k` (YOLO11n) y
   `Φ = 1` si `L` participa con al menos otro modelo, `Φ = 0.90` si solo detecta `L` y `Φ = (n/K)^β` si `L` no participa.
6. **Activacion**: solo se emite una alerta de `Accidente`/`Fire` si `ŝ >= τ_alerta` (0.35) y solo se cuenta un vehiculo si `ŝ >= τ_conteo` (0.25).
   Prioridad: Alta si `ŝ >= 0.60`, Media si `ŝ >= 0.45`, Baja en otro caso.

`γ`, `β`, `τ_alerta`, `τ_conteo` e `Ω_IoU` se ajustan en el Centro de monitoreo. La misma funcion `fuse_detections` se usa en `scripts/evaluate_fusion_test.py` para medir la fusion sobre TEST (resultados en `evaluation/fusion/test_final/`): con `ŝ >= 0.65` la precision global es 0.956.

## Limitaciones

- El conteo por IoU asume que cajas superpuestas de la misma clase corresponden al mismo objeto. Si los modelos alinean mal las cajas, el conteo puede ser inexacto.
- Una imagen aislada no permite determinar velocidad, flujo ni congestion reales.
- No se implementa seguimiento de objetos ni procesamiento de video; la arquitectura esta preparada para incorporarlos en una fase posterior.
- La agrupacion es por clase: un mismo vehiculo clasificado como `Carro` por un modelo y `Camion` por otro produce dos grupos.
- El centro de monitoreo no envia notificaciones reales ni contacta servicios de emergencia; la seccion de notificaciones es una simulacion DEMO.
- El historial de eventos se pierde al reiniciar la sesion de Streamlit, salvo que se implemente persistencia externa.

## Notas importantes

- Los checkpoints entrenados, el dataset y los resultados historicos **no deben moverse ni modificarse**.
- Los resultados mostrados son inferencias individuales por modelo; el numero de detecciones o la confianza maxima de un modelo no equivalen a una mayor precision global.
- El consenso entre modelos es una senal de apoyo, no una garantia de que el evento sea real.
- No se afirma que el sistema mida congestion real ni que envie alertas a servicios de emergencia.
