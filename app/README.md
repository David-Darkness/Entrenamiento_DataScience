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
- Imagen de evidencia con recuadros sobre eventos criticos.
- Estado general de la escena: vehiculos estimados, alertas activas y carga vehicular.
- Indicador de **carga vehicular visual** con desglose por clase.
- Alertas de posibles accidentes e incendios con prioridad, modelos que respaldan la alerta y nivel de evidencia.
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
- Se modifica algun parametro que afecta el analisis (umbral de confianza, tamano de inferencia, IoU de deduplicacion, umbrales de carga vehicular o confianza alta de un modelo).

Si se navega entre paginas, se cambia un widget o se produce un rerun de Streamlit sin cambiar la imagen ni los parametros, **no se re-ejecutan los modelos**.

Al eliminar la imagen del centro de monitoreo se limpian los resultados visuales actuales, pero se conserva el historial de eventos de la sesion.

## Reglas del motor de decision

- **Deduplicacion**: las detecciones de la misma clase provenientes de distintos modelos se agrupan mediante IoU. Cada grupo se cuenta como un unico objeto estimado.
- **Consenso para Accidente/Fire**:
  - **Alta**: el evento es detectado por dos o mas modelos con coincidencia espacial.
  - **Media**: un solo modelo detecta el evento con confianza alta (configurable, predeterminado `>= 0.70`).
  - **Baja**: deteccion aislada de baja confianza; se genera una alerta pendiente de verificacion.
- **Carga vehicular visual**: se calcula a partir de los vehiculos deduplicados (`Carro`, `Moto`, `Bus`, `Camion`) y se clasifica con umbrales configurables.

## Limitaciones

- El conteo por IoU asume que cajas superpuestas de la misma clase corresponden al mismo objeto. Si los modelos alinean mal las cajas, el conteo puede ser inexacto.
- Una imagen aislada no permite determinar velocidad, flujo ni congestion reales.
- No se implementa seguimiento de objetos ni procesamiento de video; la arquitectura esta preparada para incorporarlos en una fase posterior.
- No se presenta fusion multimodelo (WBF) como funcionalidad implementada.
- El centro de monitoreo no envia notificaciones reales ni contacta servicios de emergencia; la seccion de notificaciones es una simulacion DEMO.
- El historial de eventos se pierde al reiniciar la sesion de Streamlit, salvo que se implemente persistencia externa.

## Notas importantes

- Los checkpoints entrenados, el dataset y los resultados historicos **no deben moverse ni modificarse**.
- Los resultados mostrados son inferencias individuales por modelo; el numero de detecciones o la confianza maxima de un modelo no equivalen a una mayor precision global.
- El consenso entre modelos es una senal de apoyo, no una garantia de que el evento sea real.
- No se afirma que el sistema mida congestion real ni que envie alertas a servicios de emergencia.
