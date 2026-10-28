# Informe Técnico SmartRoad AI

**Detección inteligente de incidentes viales y vehículos mediante modelos de visión por computador**

---

## Nota editorial preliminar

Este documento es el informe académico maestro del proyecto SmartRoad AI y se construye de forma progresiva, capítulo por capítulo. En esta versión inicial se establece la estructura, las fuentes de verdad y las directrices editoriales. Los apartados técnicos se desarrollarán en entregas posteriores siguiendo las instrucciones del director del proyecto.

**Fuentes de verdad:**

1. La rama integrada `integration/four-models` y sus artefactos verificables (scripts, `args.yaml`, `results.csv`, checkpoints, matrices de confusión, curvas y registros de evaluación).
2. El informe maestro histórico del experimento inicial con el dataset antiguo:
   `C:\Users\polan\USCO\SEMESTRE SIETE 2026-2\ELECTIVA CIENCIAS DE LA COMPUTACIÓN II\Documentación\DATASET ANTIGUO\Informe_Maestro_SmartRoadAI_Modelo1_YOLOv8n_Dataset_Evaluacion.docx`.
   Sus cifras se conservan como antecedente y no se reemplazan retroactivamente.
3. Las diapositivas de la Semana 7, cuando sean proporcionadas, como guía para la elaboración e interpretación de gráficas.
4. La rúbrica académica del curso.

**Convenciones del documento:**

- `[PENDIENTE DE VERIFICACIÓN]`: dato o afirmación que aún no puede respaldarse con evidencia del repositorio.
- `[PENDIENTE DE REDACCIÓN]`: sección que requiere redacción técnica en una tarea posterior.
- Las figuras incluirán número, título, fuente exacta del artefacto e interpretación técnica.
- Las ecuaciones definirán todas sus variables y se mantendrán coherentes con la implementación real.
- Se utilizará español académico, tablas para comparaciones y referencias bibliográficas verificables.
- Logs extensos y listados de archivos se ubicarán en los anexos, no en el cuerpo principal.

---

## Índice

1. [Introducción, problema, justificación, objetivos y alcance](#capítulo-1-introducción-problema-justificación-objetivos-y-alcance)
2. [Evolución experimental: dataset inicial, experimento histórico, actualización del dataset y versión integrada](#capítulo-2-evolución-experimental)
3. [Construcción del dataset](#capítulo-3-construcción-del-dataset)
4. [Selección de modelos de Machine Learning y estado del arte](#capítulo-4-selección-de-modelos-de-machine-learning-y-estado-del-arte)
5. [Metodología experimental, simulación y análisis de resultados](#capítulo-5-metodología-experimental-simulación-y-análisis-de-resultados)
6. [Modelo matemático de las arquitecturas y del sistema conjunto](#capítulo-6-modelo-matemático-de-las-arquitecturas-y-del-sistema-conjunto)
7. [Implementación de la aplicación y arquitectura del servicio](#capítulo-7-implementación-de-la-aplicación-y-arquitectura-del-servicio)
8. [Conclusiones, limitaciones y trabajo futuro](#capítulo-8-conclusiones-limitaciones-y-trabajo-futuro)
9. [Referencias bibliográficas](#capítulo-9-referencias-bibliográficas)
10. [Anexos y evidencias complementarias](#capítulo-10-anexos-y-evidencias-complementarias)

---

## Capítulo 1. Introducción, problema, justificación, objetivos y alcance

### 1.1 Introducción

SmartRoad AI es un proyecto académico desarrollado en la asignatura Electiva Ciencias de la Computación II de la Universidad Surcolombiana, orientado al diseño y evaluación de detectores de objetos aplicados a la seguridad vial. Su motivación inicial surge de la necesidad de identificar, de manera automática, situaciones de riesgo en el tránsito —particularmente accidentes y la presencia de fuego—, con el fin de apoyar procesos de monitoreo y respuesta de emergencia. A partir de esa idea, el proyecto evolucionó hacia un estudio experimental comparativo en el que se entrenan y evalúan cuatro arquitecturas de detección: **YOLOv8n**, **YOLOv10n**, **YOLO11n** y **RT-DETR-L**.

El objeto de estudio son imágenes de escenarios viales anotadas con seis clases: **Accidente**, **Bus**, **Camion**, **Carro**, **Fire** y **Moto**. Cada clase plantea un reto diferente para un detector. Los vehículos presentan tamaños y orientaciones variables; los accidentes carecen de una forma estándar y dependen fuertemente del contexto; el fuego compite con reflejos, sombras y zonas de alto contraste cromático. Esta heterogeneidad convierte al problema en un caso de estudio útil para contrastar arquitecturas, ya que no es evidente cuál de ellas responderá mejor ante cada tipo de instancia.

El proyecto se distingue de una implementación comercial por su carácter experimental y documental. No se trata solo de obtener el detector con la mayor métrica global, sino de comprender por qué ciertas arquitecturas funcionan mejor en determinadas clases, cómo se comportan ante un dataset con desbalance de categorías y qué posibilidades ofrecen sus predicciones para una futura integración. Todos los artefactos generados —scripts, configuraciones, resultados de entrenamiento, curvas, matrices de confusión y pesos— se conservan en el repositorio como evidencia académica, de modo que el proceso pueda auditarse y reproducirse.

### 1.2 Planteamiento del problema

El problema central de SmartRoad AI puede enunciarse como sigue: *dada una imagen de un escenario vial, localizar y clasificar instancias de las seis clases definidas con la mayor precisión y robustez posible*. Resolver este problema exige más que seleccionar un modelo de detección de objetos y ajustarlo. Es necesario comprender cómo distintas arquitecturas aprenden representaciones diferentes de los datos, cómo reaccionan ante clases minoritarias o visualmente ambiguas, y si sus errores son lo suficientemente complementarios como para justificar un sistema combinado.

A partir de este problema se deriva la siguiente **pregunta de investigación**:

> *¿Cuál de las arquitecturas YOLOv8n, YOLOv10n, YOLO11n y RT-DETR-L ofrece el mejor comportamiento en la detección de cada una de las seis clases del dataset SmartRoad AI, y bajo qué condiciones sus predicciones podrían resultar complementarias para un sistema multimodelo?*

La pregunta contiene dos componentes. El primero es comparativo: se busca establecer un ranking empírico de desempeño por clase y global. El segundo es exploratorio: se indaga si la combinación de predicciones aporta ventajas respecto a los modelos individuales. Este segundo componente no asume que la integración sea beneficiosa; su utilidad debe demostrarse con evidencia, no postularse como consecuencia natural de usar varios modelos.

Una dificultad adicional reside en la trazabilidad del dataset. Si bien todos los modelos comparten la misma definición de clases, los archivos `args.yaml` registran rutas de datos distintas: YOLOv10 y RT-DETR-L apuntan al `dataset/data.yaml` local, YOLOv8 conserva una ruta absoluta de otra máquina, y YOLO11 (entrenamiento definitivo) apunta a un archivo en Google Colab. Esto impide afirmar, sin verificación adicional, que los cuatro entrenamientos consumieron exactamente la misma versión binaria del dataset. Por tanto, la comparación entre modelos debe presentarse con la debida cautela metodológica.

### 1.3 Justificación

La detección automática de accidentes y situaciones de riesgo en el tránsito tiene una relevancia social directa: reducir el tiempo de respuesta de servicios de emergencia o priorizar la atención de zonas críticas puede incidir en la gravedad de las consecuencias. Desde una perspectiva académica, SmartRoad AI justifica su desarrollo porque confronta cuatro arquitecturas representativas del estado actual de la detección de objetos con un dataset específico, en lugar de reproducir resultados genéricos de benchmarks públicos. Esta decisión metodológica permite observar cómo se comportan los modelos cuando el dominio, el balance de clases y las condiciones de anotación difieren de los escenarios estándar.

El análisis por clase es particularmente importante. Clases como *Accidente* y *Fire* tienen un alto costo de error si se clasifican incorrectamente o se omiten, pero suelen estar menos representadas que *Carro* o *Moto*. Una métrica global, como el mAP promedio, puede ocultar un mal desempeño en estas categorías críticas. Por ello, el proyecto prioriza la evaluación desagregada mediante matrices de confusión, curvas de precisión-recall y métricas de mAP por clase, además de predicciones visuales sobre subconjuntos específicos, como imágenes con presencia de fuego.

Finalmente, el estudio de una posible integración multimodelo responde a una pregunta técnica legítima: si las arquitecturas cometen errores distintos, una combinación cuidadosamente diseñada podría compensar debilidades individuales. Sin embargo, esa posibilidad no se asume como cierta. La formulación matemática del acoplamiento, lejos de ser un mero añadido teórico, sirve para delimitar bajo qué supuestos una fusión de predicciones tendría sentido y qué evidencia se requiere para validarla. De este modo, SmartRoad AI se presenta como una investigación experimental rigurosa, no como una demostración prematura de un sistema terminado.

### 1.4 Objetivo general

Diseñar, implementar y documentar un pipeline experimental de detección de objetos para seguridad vial que permita entrenar, evaluar y comparar las arquitecturas YOLOv8n, YOLOv10n, YOLO11n y RT-DETR-L sobre el dataset SmartRoad AI, y explorar las bases para una posible integración de sus predicciones.

### 1.5 Objetivos específicos

1. **Auditar y caracterizar el dataset** del proyecto, identificando el número de imágenes por split, la distribución de clases, el balance entre categorías, la presencia de anotaciones irregulares y la trazabilidad de la versión utilizada por cada modelo.

2. **Entrenar los cuatro modelos** (YOLOv8n, YOLOv10n, YOLO11n y RT-DETR-L) bajo configuraciones documentadas, respetando la estructura de carpetas del repositorio y conservando los artefactos generados como evidencia académica.

3. **Evaluar cada modelo** sobre el conjunto de prueba (*test*) del dataset, registrando métricas globales y por clase, y distinguiendo claramente entre resultados de validación durante el entrenamiento y resultados obtenidos sobre el split de TEST.

4. **Comparar el desempeño** de las cuatro arquitecturas mediante tablas, curvas y matrices de confusión, analizando las diferencias en precisión, recall, mAP@50 y mAP@50-95, con especial atención en las clases críticas.

5. **Formular el modelo matemático** de cada arquitectura y proponer un marco de acoplamiento multimodelo que defina, de manera coherente, las condiciones bajo las cuales podrían combinarse las predicciones individuales.

6. **Implementar y validar un mecanismo de integración** de los cuatro modelos, una vez que se disponga de métricas de TEST completas y comparables; hasta ese momento, esta actividad se reporta como trabajo futuro o en proceso.

7. **Documentar de forma reproducible** todo el proceso experimental, incluyendo scripts, rutas de artefactos, hiperparámetros, decisiones metodológicas y limitaciones identificadas.

**Estado de avance de los objetivos**

La auditoría del repositorio indica que los cuatro modelos han sido entrenados y que al menos uno de ellos (YOLO11n) cuenta con métricas de TEST persistidas. Sin embargo, la evaluación TEST completa de YOLOv8n, YOLOv10n y RT-DETR-L no se encuentra documentada en archivos de texto dentro del repositorio, aunque los scripts para ejecutarla existen. Asimismo, la integración multimodelo no ha sido implementada ni verificada. En consecuencia, los objetivos 3 y 6 deben considerarse parcialmente completados o pendientes de cierre, según el modelo.

### 1.6 Alcance y delimitación

**Alcance**

El proyecto abarca las siguientes actividades:

- La auditoría del dataset SmartRoad AI, incluyendo la verificación de su estructura, clases, partición en entrenamiento/validación/prueba y control de calidad de las anotaciones.
- El entrenamiento de cuatro modelos de detección de objetos (YOLOv8n, YOLOv10n, YOLO11n y RT-DETR-L) sobre el dataset disponible en el repositorio.
- La evaluación de los modelos mediante métricas estándar de detección (precision, recall, mAP@50 y mAP@50-95), tanto a nivel global como por clase, diferenciando entre resultados de validación y resultados de TEST.
- La generación de predicciones visuales y pruebas cualitativas con imágenes externas al dataset oficial.
- La formulación matemática de las arquitecturas y la exploración de un modelo de acoplamiento multimodelo.
- La documentación detallada del proceso en el presente informe técnico.

**Delimitaciones y estado actual**

- El proyecto se limita a la experimentación con el dataset almacenado en el repositorio; no incluye la recolección de nuevos datos en campo ni el despliegue de dispositivos de captura.
- Los modelos se entrenan y evalúan de manera independiente. No existe en el repositorio evidencia verificable de un sistema que cargue los cuatro checkpoints de forma simultánea, que ejecute inferencia conjunta o que fusione predicciones mediante algún algoritmo implementado.
- No se desarrolla ni se valida un servicio web, API, aplicación móvil, panel de notificaciones a emergencias, optimización de semáforos ni ningún otro sistema en producción. Cualquier referencia a estos elementos se presenta como propuesta inicial, diseño conceptual o trabajo futuro.
- No se afirma que el sistema mejore de manera demostrada la seguridad vial en un entorno real. El trabajo es de carácter académico y experimental; sus conclusiones se limitan al desempeño observado sobre el dataset y a las predicciones cualitativas generadas con imágenes externas.
- La comparación entre modelos está sujeta a la trazabilidad del dataset. Dado que los archivos de configuración registran rutas de datos distintas, no se puede garantizar sin verificación adicional que todos los entrenamientos usaron exactamente la misma versión binaria del dataset.

> **Nota editorial:** las métricas numéricas, las tablas comparativas y las figuras de desempeño se desarrollarán en los Capítulos 3, 4 y 5. El Capítulo 1 se limita a establecer el marco conceptual, la pregunta de investigación, los objetivos y las delimitaciones del proyecto.

---

## Capítulo 2. Evolución experimental

> **Propósito:** reconstruir la evolución del proyecto desde la propuesta inicial hasta la rama integrada con cuatro modelos, distinguiendo siempre la evidencia verificable de las inferencias, las discrepancias documentales y los asuntos pendientes.

### 2.1 Origen del proyecto y propuesta inicial

SmartRoad AI nació como una propuesta académica orientada a la seguridad vial. Su idea original consistía en construir un sistema de visión artificial capaz de detectar automáticamente accidentes de tránsito y focos de incendio en imágenes de escenarios viales, con el fin de agilizar la respuesta de servicios de emergencia. A partir de esa motivación, la propuesta inicial incluyó funcionalidades de alto nivel —notificaciones a emergencias, optimización del flujo de tráfico y, en algunas versiones del documento de diseño, la integración de semáforos inteligentes— que respondían más a una visión de producto futuro que a un alcance experimental verificable.

En la práctica, el desarrollo del proyecto se concentró en un objetivo más acotado y metodológicamente controlable: entrenar detectores de objetos para seis clases (`Accidente`, `Bus`, `Camion`, `Carro`, `Fire`, `Moto`) y comparar el comportamiento de distintas arquitecturas sobre un mismo dataset. Las funcionalidades de notificación, optimización de tráfico o despliegue en producción no fueron implementadas ni validadas en el repositorio; permanecen como propuestas iniciales o como trabajo futuro. Esta delimitación es importante porque evita confundir el alcance real de la evidencia experimental con las aspiraciones originales del proyecto.

### 2.2 Primeras versiones del dataset y experimentos históricos

La documentación más antigua del proyecto —registrada en `AGENTS.md`, `documentation/AUDITORIA_Y_REORGANIZACION.md` y `documentation/ESTRUCTURA_PROYECTO.md`— describe una primera versión del dataset con las siguientes cantidades:

| Partición | Imágenes | Etiquetas |
|---|---|---|
| Entrenamiento | 37.106 | 37.106 |
| Validación | 5.305 | 5.305 |
| Prueba | 2.648 | 2.648 |
| **Total** | **45.059** | **45.059** |

Esta versión del dataset fue la base del primer experimento consolidado con YOLOv8n. Según la auditoría histórica (`documentation/AUDITORIA_Y_REORGANIZACION.md`), el entrenamiento definitivo de esa etapa se ejecutó durante 20 épocas y alcanzó, sobre el conjunto de validación, las siguientes métricas:

- Precisión (P): 0,8028
- Recall (R): 0,7629
- mAP@50: 0,8240
- mAP@50-95: 0,5420

Adicionalmente, la documentación histórica del proyecto registra para ese mismo experimento una evaluación sobre el conjunto de prueba con los valores P = 0,818, R = 0,767, mAP@50 = 0,825 y mAP@50-95 = 0,548. Estas cifras deben conservarse asociadas al dataset antiguo y al modelo YOLOv8n entrenado en esa primera etapa; no deben confundirse con los resultados obtenidos posteriormente sobre el dataset actualizado.

El informe original `Informe_Maestro_SmartRoadAI_Modelo1_YOLOv8n_Dataset_Evaluacion.docx` no se encuentra en el repositorio en formato editable, por lo que los párrafos anteriores se han reconstruido a partir de los archivos de texto disponibles y de las cifras históricas comunicadas. La carpeta que hoy contiene el experimento YOLOv8n en la rama integrada se denomina `experiments/yolov8/yolov8n_20epochs-2/`; los documentos históricos la mencionan como `experiments/yolov8/entrenamiento_02_20_epochs_definitivo/`. La relación exacta entre esas dos denominaciones no ha sido verificada mediante un registro de cambios completo, pero ambas se refieren al entrenamiento de 20 épocas de YOLOv8n.

### 2.3 Actualización y auditoría del dataset actual

En octubre de 2026 se actualizó el dataset almacenado en `dataset/`. Según el equipo, el objetivo de la actualización fue mejorar el balance entre clases. El conteo actual, verificado directamente sobre el sistema de archivos del repositorio, es el siguiente:

| Partición | Imágenes | Etiquetas |
|---|---|---|
| Entrenamiento | 32.732 | 32.728 |
| Validación | 4.677 | 4.677 |
| Prueba | 2.333 | 2.333 |
| **Total** | **39.742** | **39.738** |

La diferencia respecto a la versión histórica es de −4.374 imágenes de entrenamiento, −628 de validación y −315 de prueba. [PENDIENTE DE VERIFICACIÓN: motivo exacto de la reducción y criterios de filtrado o re-etiquetado aplicados.] La pequeña diferencia entre imágenes y etiquetas en entrenamiento (4 imágenes sin etiqueta) también requiere verificación, aunque es consistente con la presencia de scripts como `dataset_check.py` y `delete_invalid_labels.py`, que sugieren que se aplicaron controles de calidad a las anotaciones.

El archivo `dataset/data.yaml` define las seis clases en el orden `Accidente`, `Bus`, `Camion`, `Carro`, `Fire`, `Moto`, y registra la fuente Roboflow Universe (`smartroadaiv1-q0ubo`, versión 1). La distribución de objetos por clase, obtenida contando las instancias anotadas en los archivos de etiquetas, muestra un desbalance importante:

| Clase | Instancias | Porcentaje |
|---|---|---|
| Carro | 51.921 | 29,21 % |
| Moto | 53.234 | 29,95 % |
| Camion | 23.929 | 13,46 % |
| Fire | 22.921 | 12,90 % |
| Accidente | 15.180 | 8,54 % |
| Bus | 10.557 | 5,94 % |
| **Total** | **177.742** | **100,00 %** |

Las clases `Carro` y `Moto` concentran casi el 60 % de las instancias, mientras que `Bus` y `Accidente` son las menos frecuentes. El dataset, aunque actualizado en octubre de 2026 con el objetivo de mejorar el balance entre clases, no está perfectamente balanceado. Este desbalance condiciona la dificultad relativa de cada clase y justifica que el análisis de desempeño se presente desagregado, no solo mediante métricas globales.

### 2.4 Entrenamiento de los cuatro modelos

La rama integrada contiene entrenamientos de cuatro arquitecturas. El equipo confirma que YOLOv8n, YOLOv10n, YOLO11n y RT-DETR-L fueron entrenados, en sus experimentos finales, con la última versión del dataset disponible en `dataset/`. Las diferencias en las rutas registradas en los `args.yaml` obedecen a distintos procedimientos de acceso: ruta absoluta local, YAML temporal generado por el script, copia de trabajo en Google Colab y ruta absoluta en otro equipo. A continuación se describe cada entrenamiento a partir de los archivos `args.yaml`, `results.csv` y los pesos disponibles, conservando las rutas originales como evidencia de trazabilidad. Salvo que se indique lo contrario, las métricas citadas corresponden al conjunto de validación registrado durante el entrenamiento.

#### 2.4.1 YOLOv8n

- **Checkpoint base:** `models/base/yolov8n.pt`
- **Experimento definitivo:** `experiments/yolov8/yolov8n_20epochs-2/`
- **Experimento interrumpido:** `experiments/yolov8/yolov8n_20epochs/` (solo contiene `args.yaml`; no hay `results.csv` ni pesos)
- **Script de entrenamiento:** `scripts/train_yolov8.py`
- **Configuración:** 20 épocas, `imgsz=640`, `batch=2`, optimizador automático (SGD con `momentum=0,937`), `seed=0`, `device=0`, `workers=0`, `amp=True`, `iou=0,7` para NMS
- **Dataset declarado en `args.yaml`:** ruta absoluta `C:/Users/omarg/Documents/GitHub/Entrenamiento_DataScience/dataset/data.yaml`, correspondiente al equipo donde se ejecutó el entrenamiento. El equipo confirma que esta ruta apuntaba a la versión actualizada del dataset.
- **Métricas finales de validación (época 20, `results.csv`):**
  - P = 0,7819
  - R = 0,7291
  - mAP@50 = 0,8005
  - mAP@50-95 = 0,5448

La mejor época registrada para ambas métricas de validación es la época 20. Sin embargo, estas cifras difieren de las reportadas en la auditoría histórica (P = 0,8028, R = 0,7629, mAP@50 = 0,8240, mAP@50-95 = 0,5420). La discrepancia puede explicarse por la reorganización de carpetas o por un reentrenamiento no documentado; no ha sido posible determinar la causa sin acceso al informe maestro original.

#### 2.4.2 YOLOv10n

- **Checkpoint base:** `models/base/yolov10n.pt`
- **Experimento:** `experiments/yolov10/entrenamiento_01_20_epochs/`
- **Script de entrenamiento:** `scripts/train_yolov10.py`
- **Configuración:** 20 épocas, `imgsz=640`, `batch=2`, optimizador automático, `seed=42`, `device=0`, `workers=0`, `amp=True`
- **Dataset:** el script crea un YAML temporal a partir de `dataset/data.yaml` (versión actualizada); el equipo confirma que este fue el dataset utilizado. Una inconsistencia documental menor es que el script aún imprime los conteos históricos (37.106 / 5.305 / 2.648) en sus mensajes de consola.
- **Métricas finales de validación (época 20, `results.csv`):**
  - P = 0,8051
  - R = 0,7587
  - mAP@50 = 0,8205
  - mAP@50-95 = 0,5401

La mejor época para ambas métricas es la época 20.

#### 2.4.3 YOLO11n

- **Checkpoint base:** `models/base/yolo11n.pt`
- **Experimento interrumpido local:** `experiments/yolov11/entrenamiento_01_20_epochs/` (contiene `args.yaml`, `labels.jpg` y batches de entrenamiento, pero no `results.csv` ni pesos)
- **Experimento definitivo:** `experiments/yolov11/entrenamiento_02_50_epochs_colab/`
- **Script de entrenamiento:** `scripts/train_yolov11.py` (configura el experimento de 20 épocas en CPU, no el entrenamiento definitivo en Colab)
- **Configuración del definitivo:** 50 épocas, `imgsz=640`, `batch=16`, optimizador `MuSGD`, `seed=0`, `device=0`, `workers=2`, `amp=True`, entrenamiento reanudado desde `/content/drive/MyDrive/SmartRoadAI/experiments/yolov11/entrenamiento_02_50_epochs_colab/weights/last.pt`
- **Dataset del definitivo:** `/content/data_colab.yaml`, preparado a partir de la versión actualizada del dataset para el entrenamiento en Google Colab. El equipo confirma que se trata de la misma versión disponible en `dataset/`, aunque el archivo exacto no está en el repositorio.
- **Métricas finales de validación (época 50, `results.csv`):**
  - P = 0,8324
  - R = 0,8049
  - mAP@50 = 0,8706
  - mAP@50-95 = 0,6223

El valor máximo de mAP@50 se observa en la época 49 (0,8707), mientras que mAP@50-95 alcanza su máximo en las épocas 49 y 50 (0,6223). El entrenamiento definitivo se ejecutó en Google Colab; el repositorio conserva sus artefactos pero no el entorno de ejecución completo.

Además, `evaluation/yolov11/test_final/metrics_summary.txt` contiene una evaluación sobre el split TEST:

| Métrica | Valor |
|---|---|
| Precisión | 0,8358 |
| Recall | 0,8068 |
| mAP@50 | 0,8716 |
| mAP@50-95 | 0,6225 |

Las métricas por clase (mAP@50-95) reportadas en ese archivo son: `Accidente` 0,7411, `Bus` 0,7311, `Camion` 0,6472, `Carro` 0,7078, `Fire` 0,3698 y `Moto` 0,5382. Estas son las únicas métricas de TEST que se encuentran persistidas como texto en el repositorio.

#### 2.4.4 RT-DETR-L

- **Checkpoint base:** `models/base/rtdetr-l.pt`
- **Experimento:** `experiments/rtdetr/entrenamiento_01_25_epochs/`
- **Script de entrenamiento:** `scripts/train_rtdetr.py`
- **Configuración:** 25 épocas, `imgsz=640`, `batch=2`, optimizador automático, `seed=0`, `device=0`, `workers=0`, `amp=True`, `deterministic=False`
- **Dataset:** ruta absoluta local `C:\...\dataset\data.yaml`, correspondiente a la versión actualizada del dataset en el equipo donde se ejecutó el entrenamiento
- **Métricas finales de validación (época 25, `results.csv`):**
  - P = 0,8315
  - R = 0,7988
  - mAP@50 = 0,8546
  - mAP@50-95 = 0,6056

La mejor época registrada para ambas métricas es la época 25. RT-DETR-L utiliza funciones de pérdida distintas a las familias YOLO (`train/giou_loss`, `train/cls_loss`, `train/l1_loss`), lo cual se refleja en las columnas de su `results.csv`.

#### Cuadro resumen de entrenamientos

| Modelo | Experimento definitivo | Épocas | Batch | Imgsz | Optimizador | Dataset declarado | P (val) | R (val) | mAP@50 (val) | mAP@50-95 (val) |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | `experiments/yolov8/yolov8n_20epochs-2/` | 20 | 2 | 640 | Auto (SGD) | Ruta absoluta Omar | 0,7819 | 0,7291 | 0,8005 | 0,5448 |
| YOLOv10n | `experiments/yolov10/entrenamiento_01_20_epochs/` | 20 | 2 | 640 | Auto (SGD) | YAML temporal local | 0,8051 | 0,7587 | 0,8205 | 0,5401 |
| YOLO11n | `experiments/yolov11/entrenamiento_02_50_epochs_colab/` | 50 | 16 | 640 | MuSGD | `/content/data_colab.yaml` | 0,8324 | 0,8049 | 0,8706 | 0,6223 |
| RT-DETR-L | `experiments/rtdetr/entrenamiento_01_25_epochs/` | 25 | 2 | 640 | Auto (SGD) | Ruta absoluta local | 0,8315 | 0,7988 | 0,8546 | 0,6056 |

Dos advertencias son obligatorias. Primera, las cuatro arquitecturas no fueron entrenadas en condiciones idénticas: difieren el número de épocas, el batch, el optimizador y el entorno (local vs. Colab). Segunda, las cifras anteriores son métricas de validación obtenidas durante el entrenamiento; por tanto, no deben interpretarse como una comparación definitiva sobre TEST, ni como una demostración de que una arquitectura supere a las demás en el escenario de prueba. La comparación interpretativa detallada de los cuatro modelos se reserva para el Capítulo 5.

### 2.5 Integración de los cuatro modelos

El término "integración" puede entenderse de tres maneras distintas, y es necesario separarlas para no sobrestimar el estado del proyecto.

1. **Integración de archivos en una misma rama de Git.** Los cuatro modelos coexisten en el repositorio bajo la estructura `experiments/<familia>/` y `evaluation/<familia>/`. Los scripts de entrenamiento y evaluación (`scripts/train_yolov8.py`, `scripts/train_yolov10.py`, `scripts/train_yolov11.py`, `scripts/train_rtdetr.py`, `scripts/test_*.py` y `scripts/predict_*.py`) también se encuentran en la misma rama. Esta integración organizacional es la que efectivamente se ha logrado.

2. **Ejecución conjunta de los cuatro modelos.** No existe en el repositorio un script que cargue los cuatro checkpoints (`best.pt` de cada familia), reciba una imagen de entrada y produzca las cuatro predicciones de forma simultánea. Cada modelo se evalúa y predice de manera independiente.

3. **Fusión efectiva de predicciones.** No se encontró una implementación verificable de ensemble, votación, fusión de cajas delimitadoras (por ejemplo, NMS ponderado o WBF) o cualquier otro mecanismo que combine las salidas de los cuatro modelos. Las carpetas `evaluation/*/external_test/results/summary.txt` contienen conteos de detecciones por clase sobre imágenes externas, pero no una métrica de desempeño combinado ni un algoritmo de fusión.

En consecuencia, la "integración" alcanzada hasta el momento es estructural y documental: los experimentos de las cuatro arquitecturas se conservan en un mismo repositorio con una convención de carpetas común. Cualquier sistema multimodelo que fusione predicciones corresponde a trabajo futuro.

### 2.6 Síntesis de la evolución experimental

La siguiente tabla resume los hitos principales del proyecto, desde la propuesta inicial hasta el estado actual de la rama integrada.

| Etapa | Periodo aproximado | Hitos | Evidencia principal | Estado |
|---|---|---|---|---|
| Propuesta inicial | Inicio del proyecto | Definición de la idea de detección de accidentes e incendios; propuesta de notificaciones y optimización de tráfico | Documento de diseño original (no disponible en el repositorio) | Propuesta parcialmente desarrollada |
| Dataset histórico | Antes de la actualización de octubre de 2026 | 45.059 imágenes (37.106 / 5.305 / 2.648); entrenamiento YOLOv8n con mAP@50 ≈ 0,824 en validación | `AGENTS.md`, `documentation/AUDITORIA_Y_REORGANIZACION.md`, `documentation/ESTRUCTURA_PROYECTO.md` | Evidencia histórica, no vigente |
| Reorganización YOLOv8 | 2026-09-24 | Limpieza de duplicados, traslado de experimentos a `experiments/yolov8/` y evaluación a `evaluation/yolov8/` | `documentation/AUDITORIA_Y_REORGANIZACION.md` | Completada |
| Actualización y balanceo del dataset | Octubre de 2026 | 39.742 imágenes (32.732 / 4.677 / 2.333); mismo esquema de clases; objetivo: mejorar el balance entre clases | Conteo directo de `dataset/` | Vigente |
| Rama integrada | Estado actual | Entrenamientos de YOLOv8n, YOLOv10n, YOLO11n y RT-DETR-L sobre el dataset actual | `experiments/*/`, `results.csv`, `args.yaml` | Entrenamientos completos; evaluaciones TEST parciales |
| Evaluación TEST | Estado actual | Métricas de TEST persistidas solo para YOLO11n; curvas y predicciones para el resto | `evaluation/yolov11/test_final/metrics_summary.txt`; carpetas `evaluation/{yolov8,yolov10,rtdetr}/test_final/` sin archivo de métricas | Parcial |
| Integración multimodelo | Estado actual | Sin script ni algoritmo de fusión verificable | `scripts/predict_*.py` individuales | Pendiente |

La evolución del proyecto presenta tres limitaciones metodológicas que afectan directamente los capítulos siguientes:

1. **Trazabilidad del dataset.** Los archivos `args.yaml` declaran rutas de datos distintas: ruta absoluta de otra máquina para YOLOv8, YAML temporal para YOLOv10, archivo en Colab para YOLO11 y ruta absoluta local para RT-DETR. El equipo confirma que estas rutas corresponden a distintos procedimientos de acceso a la misma versión actualizada del dataset. Las rutas se conservan como evidencia de trazabilidad, pero la identidad binaria exacta no puede verificarse de forma independiente en el repositorio para los entrenamientos cuyo `args.yaml` apunta a ubicaciones externas (YOLOv8 en otro equipo y YOLOv11 en `/content/data_colab.yaml`).

2. **Discrepancias en las métricas históricas de YOLOv8n.** Los valores de validación del `results.csv` actual difieren de los reportados en la auditoría histórica. La causa no ha sido determinada.

3. **Evaluaciones TEST incompletas.** Solo YOLO11n tiene métricas de TEST en un archivo de texto. Los demás modelos conservan curvas y predicciones visuales, pero no valores numéricos agregados persistidos. [PENDIENTE DE VERIFICACIÓN: ejecutar `scripts/test_yolov8.py`, `scripts/test_yolov10.py` y `scripts/test_rtdetr.py` para generar y registrar los resultados de TEST faltantes, respetando la regla de no sobrescribir evidencia existente.]

> **Nota editorial:** las conclusiones comparativas entre modelos se presentarán en el Capítulo 5, pero solo podrán considerarse robustas una vez que se disponga de métricas de TEST comparables para las cuatro arquitecturas y de una trazabilidad reproducible de las rutas de acceso al dataset.

---

## Capítulo 3. Construcción del dataset

> **Criterio de la rúbrica:** Construcción del dataset — 40 puntos.
> 
> **Recomendación técnica:** este capítulo debe demostrar que el dataset fue construido con criterios técnicos sólidos, incluyendo fuente, clases, partición, balance, preprocesamiento, control de calidad y trazabilidad.

### 3.1 Fuente y recolección de datos

El dataset de SmartRoad AI se obtuvo a partir de Roboflow Universe, proyecto `smartroadaiv1-q0ubo`, versión 1, bajo la licencia CC BY 4.0. La URL registrada en `dataset/data.yaml` es `https://universe.roboflow.com/omars-workspace-mjqor/smartroadaiv1-q0ubo/dataset/1`. Esta fuente constituye el único origen documentado de las imágenes y anotaciones; el repositorio no conserva un registro detallado de quién recolectó las imágenes originales, qué criterios de búsqueda se aplicaron ni si se realizaron transformaciones en la plataforma Roboflow antes de la exportación.

La verificación directa sobre el sistema de archivos muestra que **todas las imágenes del dataset actual tienen resolución 640 × 640 píxeles** (39.742 imágenes comprobadas). Esta uniformidad es consistente con un proceso de exportación o redimensionamiento previo, y coincide con el parámetro `imgsz=640` utilizado en los entrenamientos. Sin embargo, **no se encontró documentación que confirme el procedimiento exacto** aplicado durante la exportación (letterbox, recorte, estiramiento u otro), por lo que la afirmación de que las imágenes fueron redimensionadas permanece como una inferencia plausible, no como un hecho verificado.

Según el equipo, en **octubre de 2026** se actualizó el dataset con el objetivo de mejorar el balance entre clases. Esta actualización redujo el número total de imágenes respecto a la versión histórica (de 45.059 a 39.742), pero conservó el mismo esquema de seis clases y la misma resolución de salida.

### 3.2 Clases y taxonomía

- `nc: 6`
- `names: ['Accidente', 'Bus', 'Camion', 'Carro', 'Fire', 'Moto']`
- Mapeo IDs: 0-Accidente, 1-Bus, 2-Camion, 3-Carro, 4-Fire, 5-Moto.

El archivo `dataset/data.yaml` define seis clases en el siguiente orden:

| ID | Nombre | Descripción esperada |
|---|---|---|
| 0 | `Accidente` | Escenas o vehículos involucrados en una colisión o incidente vial |
| 1 | `Bus` | Autobuses de transporte público o privado |
| 2 | `Camion` | Vehículos de carga de tamaño mediano o grande |
| 3 | `Carro` | Automóviles particulares |
| 4 | `Fire` | Fuegos, llamas o focos de incendio en el entorno vial |
| 5 | `Moto` | Motocicletas |

La taxonomía combina cuatro clases de vehículos (`Bus`, `Camion`, `Carro`, `Moto`) con dos clases de eventos o condiciones de riesgo (`Accidente`, `Fire`). Esta organización responde a la motivación original del proyecto: detectar no solo la presencia de vehículos, sino situaciones que requieran atención de emergencia. Sin embargo, la distinción entre `Accidente` y una simple aglomeración de vehículos, o entre `Fire` y humo o reflejos, no está documentada mediante un protocolo de anotación explícito en el repositorio.

### 3.3 Partición en entrenamiento, validación y prueba

| Split | Imágenes | Etiquetas | Imágenes sin etiqueta | Etiquetas sin imagen |
|---|---|---|---|---|
| Train | 32.732 | 32.728 | 4 | 0 |
| Valid | 4.677 | 4.677 | 0 | 0 |
| Test | 2.333 | 2.333 | 0 | 0 |
| **Total** | **39.742** | **39.738** | **4** | **0** |

Las cuatro imágenes de entrenamiento que carecen de archivo de etiqueta son:

- `Electric-car-catches-fire-burns-passenger_mp4-0023_jpg.rf.0de807ada25e4de9d76e5871b4c512e6.jpg`
- `Electric-car-catches-fire-burns-passenger_mp4-0023_jpg.rf.3f62ad6f114115cab1492de5a19424a0.jpg`
- `traffic_img44_jpg.rf.168d07e5f63f9c174ddf622379490836.jpg`
- `traffic_img44_jpg.rf.76e81a6f5b6382483109f3de71347a62.jpg`

La inspección visual de estas imágenes indica que **sí contienen objetos potencialmente anotables** dentro de las clases del proyecto: las dos primeras muestran un vehículo en llamas sobre la vía (clase `Fire`, y posiblemente `Moto` o `Carro` adicionales), mientras que las dos últimas muestran un vehículo grande —aparentemente un bus o camión— fuera de la calzada en un entorno que sugiere un accidente (clases posibles `Accidente`, `Bus` o `Camion`).

En los frameworks de detección de objetos, una imagen sin anotaciones suele tratarse como un ejemplo negativo durante el entrenamiento, es decir, el modelo aprende que no hay instancias de ninguna clase en esa imagen. Sin embargo, dado el contenido visual observado, **no es posible afirmar que estas imágenes hayan sido deliberadamente dejadas como negativos ni que se trate de un error de exportación**. Ambas interpretaciones son plausibles, pero requieren confirmación del equipo o del protocolo de anotación original. [PENDIENTE DE VERIFICACIÓN: determinar si estas imágenes fueron intencionalmente excluidas del conjunto de etiquetas o si se trata de una omisión.]

En términos porcentuales, la partición por imágenes es **82,36 % train**, **11,77 % valid** y **5,87 % test**. No se encontró documentación sobre el criterio de partición (aleatoria, estratificada por clase, por fuente, por fecha, etc.) ni sobre la semilla utilizada. La siguiente tabla compara la frecuencia relativa de objetos en cada split:

| Clase | Objetos train | % train | Objetos valid | % valid | Objetos test | % test |
|---|---:|---:|---:|---:|---:|---:|
| Accidente | 11.510 | 8,25 % | 2.269 | 8,99 % | 1.401 | 10,84 % |
| Bus | 8.775 | 6,29 % | 1.203 | 4,77 % | 579 | 4,48 % |
| Camion | 18.010 | 12,90 % | 4.032 | 15,98 % | 1.887 | 14,60 % |
| Carro | 39.130 | 28,03 % | 8.199 | 32,50 % | 4.592 | 35,52 % |
| Fire | 21.198 | 15,19 % | 1.377 | 5,46 % | 346 | 2,68 % |
| Moto | 40.960 | 29,34 % | 8.151 | 32,31 % | 4.123 | 31,89 % |
| **Total** | **139.583** | **100,00 %** | **25.231** | **100,00 %** | **12.928** | **100,00 %** |

La distribución de objetos no es idéntica entre splits. En particular, `Fire` representa el 15,2 % de los objetos de entrenamiento pero solo el 2,7 % de los objetos de test, mientras que `Accidente` aumenta del 8,2 % en train al 10,8 % en test. Esto implica que la partición no está estratificada de forma perfecta por clase, lo cual debe considerarse al interpretar las métricas de TEST.

### 3.4 Balance y distribución por clase

La distribución de objetos en el dataset actual, contando únicamente las anotaciones en formato de detección válido, es la siguiente:

| Clase | Objetos | % del total |
|---|---:|---:|
| Carro | 51.921 | 29,21 % |
| Moto | 53.234 | 29,95 % |
| Camion | 23.929 | 13,46 % |
| Fire | 22.921 | 12,90 % |
| Accidente | 15.180 | 8,54 % |
| Bus | 10.557 | 5,94 % |
| **Total** | **177.742** | **100,00 %** |

Las clases `Carro` y `Moto` concentran casi el 60 % de las instancias anotadas, mientras que `Bus` y `Accidente` son las menos frecuentes. Esta distribución confirma que el dataset **no está perfectamente balanceado**, aunque la actualización de octubre de 2026 haya tenido como objetivo mejorar el balance.

Otra perspectiva relevante es el número de imágenes que contienen al menos una instancia de cada clase:

| Clase | Imágenes con la clase | % sobre imágenes etiquetadas |
|---|---:|---:|
| Carro | 18.172 | 45,73 % |
| Moto | 9.277 | 23,35 % |
| Camion | 12.399 | 31,20 % |
| Fire | 11.452 | 28,82 % |
| Accidente | 14.573 | 36,67 % |
| Bus | 8.346 | 21,00 % |

La suma de estos porcentajes supera el 100 % porque una misma imagen puede contener objetos de varias clases. La diferencia entre "objetos" e "imágenes que contienen la clase" es importante: una clase poco frecuente en número de objetos puede aparecer en muchas imágenes si cada imagen contiene pocas instancias, y viceversa.

Finalmente, la densidad media de objetos por imagen varía entre splits:

| Split | Imágenes con etiqueta | Objetos | Objetos por imagen (media) | Mediana | Mínimo | Máximo |
|---|---:|---:|---:|---:|---:|---:|
| Train | 32.728 | 139.583 | 4,26 | 3 | 1 | 58 |
| Valid | 4.677 | 25.231 | 5,39 | 4 | 1 | 54 |
| Test | 2.333 | 12.928 | 5,54 | 4 | 1 | 61 |

El conjunto de test presenta la mayor densidad media de objetos, lo cual puede influir en la dificultad relativa de la evaluación y debe tenerse en cuenta al comparar métricas entre splits.

### 3.5 Preprocesamiento y formato de anotaciones

Las anotaciones siguen el formato YOLO de detección de objetos:

```
class_id x_center y_center width height
```

donde todos los valores son numéricos, `class_id` es un entero entre 0 y 5, y las coordenadas están normalizadas al rango [0, 1] respecto al ancho y alto de la imagen. La verificación estructural del dataset actual arrojó los siguientes resultados:

- **Líneas no vacías en total:** 177.753
- **Líneas con formato de detección válido (5 valores):** 177.742
- **Líneas con formato poligonal (más de 5 valores, cantidad par de coordenadas):** 11
- **Líneas con formato incorrecto:** 0
- **Archivos de etiquetas vacíos:** 0
- **IDs de clase fuera del rango 0–5:** 0
- **Coordenadas normalizadas fuera del rango [0, 1]:** 0

Las 11 líneas poligonales están distribuidas en 11 archivos de etiquetas distintos: 10 en entrenamiento y 1 en test. Todas corresponden a la clase `Accidente` (ID 0). Esto significa que, en el dataset actual, hay 11 instancias de accidentes anotadas como polígonos en lugar de cajas delimitadoras. El repositorio incluye el script `scripts/fix_polygon_labels.py`, que convierte polígonos a bounding boxes calculando el rectángulo envolvente; sin embargo, la carpeta `dataset/labels_backup/` no existe, lo que sugiere que la conversión no fue ejecutada sobre la copia actual del dataset o que el respaldo fue eliminado. [PENDIENTE DE VERIFICACIÓN: confirmar si los modelos fueron entrenados con estas etiquetas poligonales convertidas o si el entrenador las ignoró.]

Es importante destacar que el conteo anterior corrige una cifra reportada previamente en el Capítulo 2: al contar todas las líneas no vacías de las etiquetas (incluyendo las poligonales) se obtienen 15.191 anotaciones asociadas a `Accidente`, pero al restringir el conteo a las anotaciones en formato de detección válido el total correcto es **15.180**. Esta diferencia de 11 instancias no afecta el orden de magnitud del desbalance, pero sí es relevante para la precisión del informe.

### 3.6 Control de calidad

El repositorio incluye tres scripts orientados a la auditoría y limpieza del dataset:

- `scripts/dataset_check.py`: realiza un conteo de imágenes, objetos por clase, imágenes que contienen cada clase, y validaciones estructurales (imágenes sin etiqueta, etiquetas sin imagen, IDs inválidos, coordenadas fuera de rango, líneas mal formadas).
- `scripts/fix_polygon_labels.py`: detecta anotaciones poligonales y las convierte a cajas delimitadoras, guardando una copia de seguridad en `dataset/labels_backup/`.
- `scripts/delete_invalid_labels.py`: identifica archivos de etiquetas vacíos o líneas que no cumplen el formato reconocido por `fix_polygon_labels.py`, con opción de eliminarlos.

Los resultados de ejecutar la lógica de validación sobre el dataset actual son:

- No hay archivos de etiquetas vacíos.
- No hay etiquetas sin imagen asociada.
- No hay IDs de clase inválidos.
- No hay coordenadas fuera del rango [0, 1].
- No hay líneas con cantidad de valores incorrecta, salvo las 11 líneas poligonales ya documentadas.
- Todas las imágenes tienen resolución 640 × 640 píxeles.

No se realizó una verificación exhaustiva de duplicados o casi-duplicados entre splits, por lo que **no es posible afirmar que la partición evite fugas de datos** sin una auditoría adicional.

Sin embargo, **no se encontró evidencia de que estos scripts hayan sido ejecutados de forma sistemática** ni de que sus resultados hayan quedado registrados en un log. La ausencia de `dataset/labels_backup/` y la persistencia de las 11 líneas poligonales indican que `fix_polygon_labels.py` no fue aplicado al dataset actual tal como está en el repositorio. Esto constituye un riesgo de reproducibilidad: si los entrenamientos consumieron una versión convertida del dataset, esa versión no está disponible en el árbol actual.

### 3.7 Características espaciales de las anotaciones

El tamaño y la forma de las cajas delimitadoras también varían fuertemente entre clases. Las siguientes medidas espaciales se calcularon únicamente a partir de las 177.742 anotaciones en formato de detección válido, excluyendo las 11 líneas poligonales no convertidas:

| Clase | Instancias | Ancho medio | Alto medio | Área media |
|---|---:|---:|---:|---:|
| Accidente | 15.180 | 0,479 | 0,292 | 0,172 |
| Bus | 10.557 | 0,347 | 0,375 | 0,214 |
| Camion | 23.929 | 0,182 | 0,200 | 0,055 |
| Carro | 51.921 | 0,137 | 0,114 | 0,026 |
| Fire | 22.921 | 0,202 | 0,219 | 0,080 |
| Moto | 53.234 | 0,031 | 0,094 | 0,004 |

Estos valores describen el tamaño relativo de las anotaciones en el dataset. Su interpretación como dificultad de detección debe considerarse una hipótesis metodológica, no una conclusión demostrada sin las métricas por clase de los modelos:

- `Moto` y `Carro` son las clases con objetos más pequeños en términos relativos. Desde una hipótesis metodológica, las motocicletas pueden resultar más difíciles de detectar porque presentan el área relativa más pequeña y una forma alargada; esta afirmación debe comprobarse con las métricas de desempeño por clase.
- `Bus` y `Accidente` presentan cajas relativamente grandes. La clase `Accidente` tiene un ancho medio mayor que su alto, lo cual es consistente con anotaciones que abarcan una zona amplia de la calzada; no obstante, la interpretación semántica exacta de esta clase no está documentada.
- `Fire` tiene un área media intermedia. Su forma irregular y su posible similitud visual con humo, reflejos o zonas de alto contraste la convierten, también como hipótesis, en una clase desafiante.

La distribución de la relación de aspecto (`ancho / alto`) de todas las cajas delimitadoras es:

| Relación de aspecto | Objetos | % |
|---|---:|---:|
| `< 0,5` (más alto que ancho) | 63.285 | 35,6 % |
| `0,5 – 1,0` | 51.496 | 29,0 % |
| `1,0 – 2,0` | 47.549 | 26,7 % |
| `2,0 – 4,0` | 14.795 | 8,3 % |
| `> 4,0` (más ancho que alto) | 617 | 0,3 % |

La alta proporción de objetos con relación de aspecto menor a 1 refleja la presencia de clases verticales o alargadas, especialmente `Moto`.

### 3.8 Versiones y trazabilidad

El archivo `dataset/data.yaml` es el punto central de configuración del dataset. Su contenido es:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 6
names: ['Accidente', 'Bus', 'Camion', 'Carro', 'Fire', 'Moto']

roboflow:
  workspace: omars-workspace-mjqor
  project: smartroadaiv1-q0ubo
  version: 1
  license: CC BY 4.0
  url: https://universe.roboflow.com/omars-workspace-mjqor/smartroadaiv1-q0ubo/dataset/1
```

La trazabilidad del dataset presenta tanto fortalezas como debilidades:

- **Fortalezas:** se conserva el archivo de configuración con la fuente, la versión, la licencia y el orden de las clases; las imágenes tienen resolución uniforme; las anotaciones en formato YOLO son legibles y verificables; los conteos y la estructura están documentados en el presente capítulo.
- **Debilidades:** no se dispone de un registro escrito de la actualización de octubre de 2026 (criterios de filtrado, imágenes eliminadas, decisiones de re-etiquetado); no se documenta el método de partición ni la semilla; persisten 11 anotaciones poligonales sin convertir; existen 4 imágenes de entrenamiento sin etiqueta cuyo origen no está aclarado; no se encontró un `requirements.txt`, `environment.yml` ni log de ejecución que facilite la reproducibilidad del entorno de preprocesamiento.

La siguiente tabla resume la evolución documentada del dataset:

| Versión | Periodo | Imágenes | Etiquetas | Objetos | Notas |
|---|---|---:|---:|---:|---|
| Histórica | Antes de octubre de 2026 | 45.059 | 45.059 | No verificado en repositorio | Reportada en `AGENTS.md` y `documentation/AUDITORIA_Y_REORGANIZACION.md` |
| Actual | Octubre de 2026 | 39.742 | 39.738 | 177.742 | Verificada directamente; 11 líneas poligonales y 4 imágenes train sin etiqueta |

> **Nota editorial:** los datos presentados en este capítulo se obtuvieron directamente del sistema de archivos del repositorio. Cualquier afirmación sobre procedimientos de recolección, filtrado o anotación que no esté respaldada por un archivo ejecutable, un log o una documentación escrita se ha marcado como pendiente de verificación. Estas limitaciones deben considerarse al interpretar los resultados de los modelos en los capítulos siguientes.

---

## Capítulo 4. Selección de modelos de Machine Learning y estado del arte

> **Criterio de la rúbrica:** Selección de modelos ML — 50 puntos.
> 
> **Recomendación técnica:** se espera una revisión del estado del arte en detección de objetos para seguridad vial, la justificación de la elección de YOLOv8, YOLOv10, YOLO11 y RT-DETR-L, y la explicación de las variantes utilizadas.

### 4.1 Detección de objetos en seguridad vial

La detección de objetos en escenarios de tránsito busca localizar y clasificar, en una imagen o secuencia de video, a los actores y eventos relevantes para la seguridad vial: vehículos, motocicletas, autobuses, accidentes, incendios, peatones y señales de tránsito [1][2]. Desde la aparición de las arquitecturas basadas en aprendizaje profundo, el campo ha dejado de depender de descriptores artificiales (Haar, HOG, SIFT) para adoptar redes convolucionales (CNN) capaces de aprender representaciones jerárquicas directamente de los datos [3]. Esta transición permitió mejorar drásticamente la precisión en condiciones adversas como baja iluminación, oclusión parcial, cambios de escala y fondos complejos, situaciones frecuentes en cámaras de tráfico urbano e interurbano.

En el contexto de SmartRoad AI, el problema se acota a seis clases (`Accidente`, `Bus`, `Camion`, `Carro`, `Fire`, `Moto`) que combinan vehículos y eventos de riesgo. Las aplicaciones prácticas de un detector con esta taxonomía incluyen: monitoreo automático de incidentes en cámaras de peaje y semáforos, generación de alertas tempranas de incendio vehicular, y conteo/clasificación de flujo vehicular para la gestión de tráfico [1]. Estos escenarios imponen dos restricciones de diseño: (i) la capacidad de operar en tiempo real o cerca de tiempo real, y (ii) la robustez frente a clases poco frecuentes pero críticas, como accidentes e incendios, que suelen estar subrepresentadas en los datasets [2].

### 4.2 Familia YOLO: evolución y características

La familia YOLO (*You Only Look Once*) definió el paradigma de detectores de una sola etapa: una sola pasada por la red predice simultáneamente la clase y la ubicación de todos los objetos de la imagen, lo que reduce la latencia respecto a los detectores de dos etapas (R-CNN, Fast R-CNN, Faster R-CNN) [4]. Las versiones posteriores han mejorado sucesivamente la relación velocidad–precisión, la escalabilidad a dispositivos de borde y la facilidad de entrenamiento.

**YOLOv8** (Ultralytics, 2023) introduce una cabeza *anchor-free* dividida (*decoupled head*), un backbone y un neck rediseñados basados en CSPDarknet, y una API unificada para detección, segmentación, clasificación, pose y *oriented bounding boxes* [5]. La eliminación de *anchors* predeterminados simplifica el ajuste de hiperparámetros y reduce la dependencia de la forma estadística de los objetos en el conjunto de entrenamiento. YOLOv8 se convirtió en una referencia estable y ampliamente documentada para proyectos de detección en tiempo real, por lo que su inclusión en SmartRoad AI permite comparar contra una línea base consolidada.

**YOLOv10** (Wang et al., 2024) avanza hacia la detección *end-to-end* sin NMS (*non-maximum suppression*) mediante el mecanismo de *consistent dual assignments* y un diseño del modelo guiado simultáneamente por eficiencia y precisión [6]. Al evitar la post-procesación por NMS, YOLOv10 reduce la latencia de inferencia y facilita el despliegue en pipelines de bajo costo computacional. Además, los experimentos reportados en COCO muestran que YOLOv10 mejora la frontera rendimiento–eficiencia con respecto a generaciones anteriores y a RT-DETR a escala similar [6].

**YOLO11** (Ultralytics, 2024) reemplaza los módulos C2f por bloques C3k2 y añade el módulo C2PSA (*Cross-Stage Partial Spatial Attention*) en el backbone, mejorando la extracción de características espaciales y el manejo de oclusiones sin un crecimiento desproporcionado del número de parámetros [7]. Según la documentación oficial, YOLO11m alcanza mayor precisión en COCO que YOLOv8m con un 22 % menos de parámetros [7]. Su inclusión permite evaluar si las mejoras de atención espacial se traducen en ganancias sobre las clases pequeñas del dataset SmartRoad AI, especialmente `Moto`.

Las variantes *nano* (`n`) de cada generación fueron elegidas para mantener un modelo compacto y rápido, coherente con la posibilidad futura de despliegue en dispositivos de borde o en cámaras con recursos limitados.

### 4.3 Transformers para detección: DETR y RT-DETR

Carion et al. (2020) propusieron DETR (*Detection Transformer*), el primer detector *end-to-end* basado exclusivamente en transformadores [8]. DETR utiliza un encoder transformer para procesar características de la imagen, un decoder transformer para refinar un conjunto fijo de *queries* de objeto, y una función de costo de emparejamiento bipartita (*bipartite matching*) para asignar predicciones a ground-truth. Esta formulación elimina la necesidad de *anchors*, regresores de propuesta y NMS, simplificando conceptualmente el pipeline de inferencia.

El principal inconveniente de DETR original es su alto costo computacional y memoria, que dificulta su uso en aplicaciones en tiempo real. Para superar esta limitación, Zhao et al. (2024) presentaron RT-DETR (*Real-Time Detection Transformer*) [9]. RT-DETR incorpora un *encoder híbrido eficiente* que separa la interacción intraescala de la fusión interescala, una selección de *queries* consciente de la incertidumbre (*uncertainty-minimal query selection*) para mejorar la calidad de las consultas iniciales, y la posibilidad de ajustar la velocidad variando el número de capas del decoder sin reentrenar. En COCO val2017, RT-DETR-L alcanza 53,0 % AP a 114 FPS en GPU T4, superando a los detectores YOLO contemporáneos de escala comparable [9]. Esta combinación de precisión y ausencia de NMS justifica su inclusión como contraparte transformer del experimento.

### 4.4 Criterios de selección para SmartRoad AI

La elección de cuatro modelos para SmartRoad AI responde a cuatro criterios principales:

1. **Cobertura del espectro velocidad–precisión.** Las tres variantes YOLO nano representan el estado del arte en detección en tiempo real con bajo consumo computacional, mientras que RT-DETR-L explora el extremo de mayor precisión sin post-procesamiento NMS.
2. **Relevancia para el dataset.** El dataset contiene seis clases, algunas pequeñas y alargadas (`Moto`) y otras amplias o irregulares (`Accidente`, `Fire`). Comparar arquitecturas con diferentes mecanismos de atención y asignación permite observar cuál se adapta mejor a estas formas.
3. **Acceso a pesos preentrenados y facilidad de implementación.** Todos los modelos están disponibles en la API de Ultralytics, lo que garantiza un pipeline de entrenamiento, validación y exportación homogéneo.
4. **Base de comparación temporal.** YOLOv8 es la generación anterior consolidada; YOLOv10 introduce entrenamiento sin NMS; YOLO11 incorpora atención espacial; RT-DETR-L representa la línea transformer optimizada para tiempo real. Juntos permiten caracterizar el margen de mejora disponible entre 2023 y 2024.

No se implementó un ensemble o fusión de predicciones entre modelos en el código disponible; la selección se limita a entrenar y evaluar cada arquitectura de manera independiente.

### 4.5 Modelos seleccionados y variantes exactas

| Modelo | Variante | Peso base | Script de entrenamiento |
|---|---|---|---|
| YOLOv8 | YOLOv8n | `models/base/yolov8n.pt` | `scripts/train_yolov8.py` |
| YOLOv10 | YOLOv10n | `models/base/yolov10n.pt` | `scripts/train_yolov10.py` |
| YOLO11 | YOLO11n | `models/base/yolo11n.pt` | `scripts/train_yolov11.py` |
| RT-DETR | RT-DETR-L | `models/base/rtdetr-l.pt` | `scripts/train_rtdetr.py` |

La columna "Variante" indica el tamaño exacto del modelo entrenado en el experimento (`n` = *nano*, la versión más ligera de cada familia). La columna "Peso base" señala el checkpoint preentrenado en COCO utilizado como punto de partida, y "Script de entrenamiento" el archivo de Python que orquesta la llamada a `model.train()` dentro del proyecto.

---

## Capítulo 5. Metodología experimental, simulación y análisis de resultados

> **Criterio de la rúbrica:** Simulación de resultados — 50 puntos.
> 
> **Recomendación técnica:** este capítulo debe describir el entorno experimental, el pipeline de entrenamiento, las métricas utilizadas y presentar los resultados de forma distinguible por tipo (entrenamiento, validación, TEST, pruebas externas).

### 5.1 Entorno de desarrollo y reproducibilidad

El proyecto se desarrolla en **Python** sobre el framework **PyTorch**, utilizando la biblioteca **Ultralytics** como interfaz unificada para entrenar, validar y ejecutar inferencia con los modelos YOLOv8, YOLOv10, YOLO11 y RT-DETR. Los entrenamientos locales especifican `device='0'`, lo que indica una GPU NVIDIA disponible en la máquina de desarrollo. El entrenamiento de YOLO11n se ejecutó en **Google Colab** (`device='0'`, ruta `/content/...`), lo que sugiere el uso de una GPU proporcionada por el entorno cloud.

No se encontró en el repositorio un archivo `requirements.txt`, `environment.yml`, `pyproject.toml` ni un log de instalación que documente las versiones exactas de Python, PyTorch, CUDA, cuDNN ni Ultralytics. Esta ausencia constituye un riesgo de reproducibilidad: sin fijar estas versiones, los resultados podrían variar al reejecutar el proyecto en otro entorno. Los `args.yaml` de cada experimento sí conservan la configuración interna de Ultralytics (por ejemplo, `amp: true`, `deterministic: true`, `mosaic: 1.0`, `close_mosaic: 10`), pero no la versión del paquete.

### 5.2 Pipeline general de entrenamiento

El pipeline seguido por los cuatro modelos es homogéneo gracias a la API de Ultralytics:

1. **Carga del modelo base** desde `models/base/`.
2. **Configuración del dataset** a través de `data.yaml` (local, temporal o en Colab).
3. **Entrenamiento** con `model.train(...)`, especificando `project`, `name`, `epochs`, `imgsz`, `batch`, `device`, `seed`, `optimizer`, etc.
4. **Validación interna** al final de cada época sobre el split de validación.
5. **Evaluación final sobre TEST** con `model.val(split="test")`.
6. **Predicciones visuales y pruebas con imágenes externas** mediante `model.predict(...)`.

Todos los scripts fijan `imgsz=640`, `iou=0.7` y una confianza mínima de `0.25` para las predicciones externas. La única excepción sustancial es YOLO11n, cuyo entrenamiento se reanudó (`resume: true`) desde un checkpoint intermedio en Google Drive, con `exist_ok: true`.

### 5.3 Configuración experimental por modelo

| Modelo | Épocas | imgsz | batch | Dispositivo | Optimizador | Seed | Data YAML |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 20 | 640 | 2 | GPU local (`device=0`) | auto | 0 | `C:/Users/omarg/.../Entrenamiento_DataScience/dataset/data.yaml` |
| YOLOv10n | 20 | 640 | 2 | GPU local (`device=0`) | auto | 42 | YAML temporal (`AppData/Local/Temp/data_gtr_vqp5_smartroad_yolov10.yaml`) |
| YOLO11n | 50 | 640 | 16 | GPU Colab (`device=0`) | MuSGD | 0 | `/content/data_colab.yaml` (reanudado desde Google Drive) |
| RT-DETR-L | 25 | 640 | 2 | GPU local (`device=0`) | auto | 0 | `dataset/data.yaml` |

Tres aspectos de esta tabla son importantes para la interpretación posterior:

- **Dataset binario no uniforme:** YOLOv8n apunta a una ruta en el equipo de Omar, YOLO11n a una copia en Colab (`/content/data_colab.yaml`) y YOLOv10n/RT-DETR-L a una copia local. El equipo afirma que todos usaron la versión actual del dataset, pero el repositorio no permite verificarlo binariamente.
- **Batch size diferente:** YOLO11n usó batch 16, mientras que los modelos locales usaron batch 2. Esto afecta la estimación de estadísticas de batch normalization y la estabilidad del gradiente.
- **Reanudación:** YOLO11n se entrenó con `resume: true` desde `last.pt`. Los demás modelos partieron de cero desde el peso base preentrenado en COCO.

### 5.4 Métricas de evaluación

Las métricas empleadas son las estándar en detección de objetos:

- **IoU (*Intersection over Union*):** razón entre el área de intersección y el área de unión de una caja predicha y su ground-truth. Una detección se considera verdadero positivo cuando IoU ≥ umbral (típicamente 0.5).
- **Precision:** proporción de detecciones predichas que son correctas, \(Precision = TP / (TP + FP)\).
- **Recall:** proporción de objetos reales detectados, \(Recall = TP / (TP + FN)\).
- **mAP@50:** *mean Average Precision* promediado sobre las seis clases, usando IoU ≥ 0.5 como criterio de acierto.
- **mAP@50-95:** promedio de mAP calculado a 10 umbrales de IoU entre 0.50 y 0.95, con paso 0.05. Es una métrica más estricta que refleja la calidad de localización.

Es fundamental distinguir entre **métricas de validación** (calculadas por Ultralytics al final de cada época sobre el split `valid`) y **métricas de TEST** (calculadas una vez terminado el entrenamiento con `split="test"`). En un informe académico no deben mezclarse: la validación guía el ajuste del modelo, mientras que TEST mide la generalización sobre datos no vistos durante el entrenamiento ni la selección de hiperparámetros.

### 5.5 Resultados de entrenamiento y validación

La tabla siguiente resume las métricas finales de validación extraídas de la última fila de cada `results.csv`:

| Modelo | Épocas | P (val) | R (val) | mAP@50 (val) | mAP@50-95 (val) | Pérdidas finales (box / cls / dfl) |
|---|---:|---:|---:|---:|---:|---|
| YOLOv8n | 20 | 0,782 | 0,729 | 0,801 | 0,545 | 1,096 / 0,918 / 1,190 |
| YOLOv10n | 20 | 0,805 | 0,759 | 0,820 | 0,540 | 1,300 / 0,736 / 1,236 |
| YOLO11n | 50 | 0,832 | 0,805 | 0,871 | 0,622 | 1,011 / 0,665 / 1,052 |
| RT-DETR-L | 25 | 0,831 | 0,799 | 0,855 | 0,606 | 0,353 / 0,437 / 0,150* |

\* Las pérdidas de RT-DETR-L corresponden a `giou_loss`, `cls_loss` y `l1_loss`, respectivamente, ya que esta arquitectura no utiliza `dfl_loss`.

Todas las curvas de pérdida descendieron de forma estable a lo largo del entrenamiento, sin signos evidentes de divergencia. YOLO11n obtuvo la mejor validación en ambas métricas de mAP, seguido de cerca por RT-DETR-L. YOLOv10n superó ligeramente a YOLOv8n en mAP@50, aunque ambos modelos de 20 épocas quedaron por debajo de YOLO11n y RT-DETR-L.

### 5.6 Resultados de evaluación sobre TEST

Solo YOLO11n tiene un archivo de métricas de TEST persistido:

| Métrica | Valor |
|---|---:|
| Precision | 0,8358 |
| Recall | 0,8068 |
| mAP@50 | 0,8716 |
| mAP@50-95 | 0,6225 |

El desempeño por clase (mAP@50-95) sobre TEST fue:

| Clase | mAP@50-95 |
|---|---:|
| Accidente | 0,7411 |
| Bus | 0,7311 |
| Camion | 0,6472 |
| Carro | 0,7078 |
| Fire | 0,3698 |
| Moto | 0,5382 |

La clase `Fire` presenta el valor más bajo (0,3698), lo que sugiere que, a pesar de contar con más de 22.000 instancias de entrenamiento, la localización precisa o la separación visual de fuego frente a fondos similares sigue siendo un desafío. `Accidente` y `Bus` obtienen los mejores resultados por clase, mientras que `Moto` y `Camion` quedan en niveles intermedios.

Para YOLOv8n, YOLOv10n y RT-DETR-L se encontraron las carpetas `evaluation/{yolov8,yolov10,rtdetr}/test_final/` con curvas y batch de predicción, pero **no se encontró un archivo de métricas numéricas persistido** equivalente a `metrics_summary.txt`. Por tanto, sus resultados de TEST no pueden reportarse en este informe sin ejecutar nuevamente la evaluación o recuperar la salida impresa de los scripts originales. [PENDIENTE DE VERIFICACIÓN]

### 5.7 Pruebas con imágenes externas

Se dispone de 25 imágenes en `external_data/images/`, sin etiquetas ground-truth. Sobre ellas se ejecutaron inferencias con YOLOv8n, YOLOv10n y RT-DETR-L. Los resúmenes cualitativos son:

| Modelo | Imágenes procesadas | Detecciones totales | Accidente | Bus | Camion | Carro | Fire | Moto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 19 | 34 | 7 | 2 | 1 | 9 | 5 | 10 |
| YOLOv10n | 19 | 50 | 4 | 5 | 0 | 11 | 4 | 26 |
| RT-DETR-L | 25 | 303 | 11 | 33 | 31 | 111 | 6 | 111 |

Hay dos inconsistencias que deben aclararse:

1. **Número de imágenes procesadas:** YOLOv8n e YOLOv10n reportan 19 imágenes, mientras que la carpeta actual contiene 25. Esto puede deberse a que, en el momento de ejecutar las pruebas, la carpeta tenía menos imágenes, o a que el script filtró algunas extensiones. Dado que el script actual acepta `.jpeg`, todas las imágenes actuales deberían procesarse.
2. **Checkpoint de YOLOv8n:** el resumen externo de YOLOv8n indica que usó `experiments/yolov8/entrenamiento_02_20_epochs_definitivo/weights/best.pt`, una ruta que **no existe en el repositorio actual**. El script actual apunta a `experiments/yolov8/yolov8n_20epochs-2/weights/best.pt`. Esto indica que el resumen externo proviene de una ejecución anterior o de una rama distinta.

Además, RT-DETR-L generó 303 detecciones sobre 25 imágenes (promedio de 12,1 por imagen), muy por encima de YOLOv8n (1,8 por imagen) y YOLOv10n (2,6 por imagen). Sin etiquetas de referencia no es posible determinar si este comportamiento refleja mayor sensibilidad o un exceso de falsos positivos; por eso estas pruebas se reportan solo como exploración cualitativa de generalización.

### 5.8 Análisis comparativo

Desde la perspectiva de las métricas de validación disponibles, el orden de desempeño es:

1. **YOLO11n** — mejor mAP@50 (0,871) y mAP@50-95 (0,622).
2. **RT-DETR-L** — segundo en ambas métricas (0,855 y 0,606), con una arquitectura completamente distinta.
3. **YOLOv10n** — mAP@50 0,820, ligeramente por encima de YOLOv8n.
4. **YOLOv8n** — mAP@50 0,801, mAP@50-95 0,545.

En TEST, solo YOLO11n puede afirmar una métrica global: mAP@50 = 0,8716 y mAP@50-95 = 0,6225. Estos valores son consistentes con su validación, lo que indica que el modelo no sufrió sobreajuste severo respecto al split de validación.

No obstante, **esta comparación tiene limitaciones metodológicas importantes**:

- Los modelos no fueron entrenados con exactamente los mismos hiperparámetros: YOLO11n tuvo 50 épocas y batch 16, mientras que los demás usaron 20–25 épocas y batch 2.
- La versión binaria exacta del dataset no es idéntica en todos los `args.yaml`: YOLOv8n apunta a una ruta externa, YOLO11n a Colab, y los otros dos a copias locales.
- No se dispone de métricas de TEST para tres de los cuatro modelos.

Por estas razones, el ranking anterior debe interpretarse como una **comparación explorativa** de arquitecturas bajo condiciones aproximadas, no como una evaluación estrictamente controlada. Un experimento completamente reproducible requeriría entrenar los cuatro modelos desde el mismo dataset, con el mismo batch size, mismas épocas y una semilla fijada, y persistir explícitamente las métricas de TEST.

---

## Capítulo 6. Modelo matemático de las arquitecturas y del sistema conjunto

> **Criterio de la rúbrica:** Modelo matemático — 100 puntos.
> 
> **Recomendación técnica:** se espera la formulación matemática de las arquitecturas, las funciones de pérdida, las métricas y, si aplica, el modelo de acoplamiento de los cuatro detectores.

### 6.1 Fundamentos matemáticos de YOLO

La familia YOLO trata la detección como un problema de regresión y clasificación simultáneas sobre una imagen de entrada \(I \in \mathbb{R}^{H \times W \times 3}\). La red se divide conceptualmente en tres partes:

1. **Backbone:** extrae características jerárquicas a diferentes resoluciones espaciales.
2. **Neck:** fusiona las características de diferentes escalas (típicamente mediante FPN/PAN o variantes como CSP).
3. **Head:** produce, para cada celda de la grilla de salida y cada escala, un vector de predicción.

En su formulación ancla-libre (*anchor-free*), como la empleada por YOLOv8, YOLOv10 y YOLO11, la cabeza predice directamente un conjunto de valores por objeto candidato:

\[
\mathbf{o} = (t_x,\, t_y,\, t_w,\, t_h,\, p_1,\, p_2,\, \dots,\, p_C),
\]

donde \((t_x, t_y, t_w, t_h)\) codifican la caja delimitadora y \((p_1,\dots,p_C)\) las probabilidades de clase. Si \(s\) es el paso (*stride*) de la característica y \((i,j)\) son los índices de la celda, las coordenadas normalizadas de la caja predicha se obtienen como:

\[
\begin{aligned}
b_x &= s\left(i + \sigma(t_x)\right), \\
b_y &= s\left(j + \sigma(t_y)\right), \\
b_w &= s\, e^{t_w}, \\
b_h &= s\, e^{t_h},
\end{aligned}
\]

con \(\sigma(\cdot)\) la función sigmoide. La clase predicha es \(\hat{c} = \arg\max_c \, p_c\).

Durante el entrenamiento, las predicciones se emparejan con los ground-truth mediante una estrategia de asignación (en YOLOv8 se usa asignación dinámica basada en IoU y costo de clasificación). La función de pérdida combina tres términos:

\[
\mathcal{L}_{YOLO} = \lambda_{box}\, \mathcal{L}_{box} + \lambda_{cls}\, \mathcal{L}_{cls} + \lambda_{dfl}\, \mathcal{L}_{dfl}.
\]

- \(\mathcal{L}_{box}\): pérdida de localización, comúnmente CIoU (*Complete Intersection over Union*) o DFL (*Distribution Focal Loss*) en las versiones más recientes.
- \(\mathcal{L}_{cls}\): entropía cruzada binaria (*binary cross-entropy*) entre las probabilidades predichas y la distribución objetivo.
- \(\mathcal{L}_{dfl}\): focaliza el aprendizaje de los cuantiles de la distribución espacial de los bordes de la caja.

En la inferencia, las predicciones crudas se filtran por umbral de confianza y, salvo en YOLOv10 que entrena sin NMS, se aplica **NMS** para eliminar cajas redundantes:

\[
s_i \leftarrow \begin{cases}
s_i, & \text{IoU}(B_i, B_{max}) < \tau_{NMS} \\[6pt]
0, & \text{en otro caso},
\end{cases}
\]

donde \(B_{max}\) es la caja de mayor confianza y \(\tau_{NMS}\) (por ejemplo, 0.7) el umbral de superposición permitido.

### 6.2 Fundamentos matemáticos de RT-DETR

RT-DETR reformula la detección como un problema de *set prediction*. Dada una imagen \(I\), un backbone CNN produce un mapa de características de menor resolución. Un **encoder híbrido** transforma dichas características en una secuencia de embeddings de imagen, desacoplando la interacción intraescala de la fusión interescala. A continuación, un **decoder transformer** recibe un conjunto fijo de *queries* aprendibles \(\mathbf{q} = \{q_1, \dots, q_N\}\) y los refina iterativamente en capas cruzadas de *self-attention* y *cross-attention* con los embeddings de imagen.

La salida de cada capa del decoder es una predicción \(\hat{y}_i = (\hat{c}_i, \hat{B}_i)\), donde \(\hat{c}_i\) es la distribución sobre clases y \(\hat{B}_i\) las coordenadas de la caja. Para entrenar sin *anchors* y sin NMS, DETR utiliza un **emparejamiento bipartita** mediante el algoritmo húngaro. Dado un conjunto de ground-truth \(Y = \{y_1, \dots, y_M\}\) (con \(M \le N\)) y predicciones \(\hat{Y} = \{\hat{y}_1, \dots, \hat{y}_N\}\), se busca la permutación \(\sigma^*\) que minimice el costo total:

\[
\sigma^* = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{M} \mathcal{C}_{match}(y_i, \hat{y}_{\sigma(i)}) + \sum_{i=M+1}^{N} \mathcal{C}_{no\text{-}obj}(\hat{y}_{\sigma(i)}).
\]

El costo de emparejamiento \(\mathcal{C}_{match}\) combina costo de clasificación y costo de caja:

\[
\mathcal{C}_{match}(y_i, \hat{y}_j) = -\hat{p}_{\sigma(j)}(c_i) + \lambda_{box}\, \mathcal{L}_{box}(B_i, \hat{B}_j) + \lambda_{cls}\, \mathcal{L}_{cls}(c_i, \hat{c}_j),
\]

donde la primera suma penaliza la probabilidad asignada a la clase correcta y los demás términos penalizan el error de localización.

La pérdida total de RT-DETR se expresa como:

\[
\mathcal{L}_{DETR} = \sum_{i=1}^{M} \Big[ \lambda_{cls}\, \mathcal{L}_{cls}(c_i, \hat{c}_{\sigma^*(i)}) + \lambda_{box}\, \mathcal{L}_{box}(B_i, \hat{B}_{\sigma^*(i)}) \Big],
\]

donde \(\mathcal{L}_{box}\) suele ser una combinación de pérdida L1 y GIoU, y \(\mathcal{L}_{cls}\) entropía cruzada. RT-DETR elimina por diseño la etapa de NMS, lo que simplifica el despliegue a costa de un mayor consumo de memoria en el decoder.

### 6.3 Funciones de pérdida y métricas

A continuación se formalizan las funciones y métricas utilizadas en los experimentos.

**IoU.** Dadas dos cajas \(A\) y \(B\):

\[
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}.
\]

**GIoU.** Generaliza IoU para cajas no solapadas:

\[
\text{GIoU}(A, B) = \text{IoU}(A, B) - \frac{|C \setminus (A \cup B)|}{|C|},
\]

donde \(C\) es la caja envolvente mínima de \(A\) y \(B\).

**CIoU.** Incorpora solapamiento, distancia entre centroides y relación de aspecto:

\[
\text{CIoU}(A, B) = \text{IoU}(A, B) - \frac{\rho^2(A_{ctr}, B_{ctr})}{d^2} - \alpha \, v,
\]

con \(\rho\) la distancia euclídea entre centroides, \(d\) la diagonal de \(C\), \(\alpha\) un factor de equilibrio y \(v\) un término de relación de aspecto.

**Pérdida de caja.** Tanto YOLO (CIoU) como RT-DETR (GIoU/L1) definen:

\[
\mathcal{L}_{box} = 1 - \text{IoU-based}(A, B).
\]

**Entropía cruzada binaria para clasificación.** Para una clase \(c\) con etiqueta \(y_c \in \{0,1\}\):

\[
\mathcal{L}_{cls} = -\sum_{c=1}^{C} \Big[ y_c \log(\hat{p}_c) + (1 - y_c) \log(1 - \hat{p}_c) \Big].
\]

**Average Precision (AP).** Para una clase, AP es el área bajo la curva Precision–Recall:

\[
\text{AP} = \int_0^1 p(r)\, dr,
\]

donde \(p(r)\) es la precisión máxima alcanzada para un recall mayor o igual a \(r\). En la práctica se interpola en 101 puntos.

**mAP.** Promedio de AP sobre todas las clases del problema:

\[
\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c.
\]

Las variantes mAP@50 y mAP@50-95 difieren en el umbral de IoU exigido para considerar un acierto.

### 6.4 Modelo de acoplamiento multimodelo

El repositorio incluye el documento `documentation/MARCO_TEORICO_MODELO_MATEMATICO.md`, que propone un **ensamble heterogéneo de cuatro detectores** basado en *Weighted Boxes Fusion* (WBF) [10] con un esquema de confianza asimétrica. A continuación se resume la formulación teórica; **no se encontró código ejecutable en la rama integrada que implemente este acoplamiento**, por lo que debe considerarse un modelo propuesto, no una funcionalidad operativa.

Dada una imagen \(I\), cada modelo \(k \in \{1,2,3,4\}\) produce un conjunto de detecciones:

\[
\mathcal{D}_k = \{(B_{k,i},\, s_{k,i},\, c_{k,i})\}_{i=1}^{N_k},
\]

donde \(B\) es la caja, \(s\) la confianza y \(c\) la clase. El procedimiento teórico consta de cuatro pasos:

1. **Ponderación por modelo.** A partir de una métrica de calidad \(P_k\) (por ejemplo, mAP@50), se calcula un peso:

\[
w_k = \frac{P_k^\gamma}{\sum_{j=1}^{4} P_j^\gamma}, \qquad \gamma \ge 1.
\]

2. **Agrupamiento espacial.** Se construye un grafo de detecciones cuyos nodos se conectan si \(\text{IoU}(B_a, B_b) \ge \Omega_{IoU}\). Cada componente conexa define un clúster \(\mathcal{C}_m\) correspondiente a un único objeto.

3. **Fusión geométrica.** La caja consolidada se obtiene como promedio ponderado por la autoridad del modelo y la confianza local:

\[
\hat{B}_m = \frac{\sum_{t \in \mathcal{C}_m} w_{k(t)}\, s_t\, B_t}{\sum_{t \in \mathcal{C}_m} w_{k(t)}\, s_t}.
\]

4. **Fusión de confianza.** La confianza final se ajusta por un factor de consenso asimétrico:

\[
\hat{s}_m = \bar{s}_m \cdot \Phi(\mathcal{C}_m),
\qquad
\bar{s}_m = \frac{\sum_{t \in \mathcal{C}_m} w_{k(t)}\, s_t}{\sum_{t \in \mathcal{C}_m} w_{k(t)}},
\]

donde \(\Phi(\mathcal{C}_m)\) penaliza los clústeres en los que no participa el modelo con mayor precisión teórica (RT-DETR-L en la propuesta).

Una detección fusionada se considera válida si \(\hat{s}_m \ge \tau_{conf}\). Los parámetros \(\gamma\), \(\Omega_{IoU}\), \(\beta\) y \(\tau_{conf}\) quedan como hiperparámetros del sistema de alertas.

### 6.5 Coherencia entre modelo matemático e implementación

La formulación matemática anterior describe lo que los modelos **deberían** hacer o, en el caso del ensamble, lo que el equipo **propuso** hacer. Sin embargo, la inspección del código disponible revela que:

- Cada modelo se entrena y evalúa de forma **independiente**; no existe un script que cargue los cuatro checkpoints simultáneamente.
- No se encontró implementación del agrupamiento IoU, la fusión ponderada ni la función de confianza asimétrica descrita en el Marco Teórico.
- Las métricas reportadas provienen directamente de Ultralytics (`results.csv` y `model.val`), sin post-procesamiento por ensamble.

Por tanto, el **modelo de acoplamiento multimodelo es teórico** en el estado actual del repositorio. Cualquier afirmación sobre mejoras de precisión obtenidas por fusión de los cuatro detectores requeriría, como mínimo, un script ejecutable y una evaluación sobre TEST que compare el ensamble contra cada modelo individual.

---

## Capítulo 7. Implementación de la aplicación y arquitectura del servicio

> **Criterio de la rúbrica:** Implementación de aplicación — 50 puntos.
> 
> **Recomendación técnica:** describir la arquitectura de software, los scripts de entrenamiento/evaluación/predicción, el flujo de datos y, si existe, el servicio de inferencia. No describir funcionalidades no implementadas.

### 7.1 Estructura del repositorio

El repositorio sigue una organización orientada a separar datos, código, artefactos de entrenamiento, evaluaciones y documentación:

| Carpeta | Rol |
|---|---|
| `dataset/` | Imágenes y anotaciones del dataset SmartRoad AI. Es de **solo lectura**; no debe modificarse. |
| `scripts/` | Código Python para entrenamiento, evaluación, predicción y auditoría del dataset. |
| `models/base/` | Pesos preentrenados descargados (YOLOv8n, YOLOv10n, YOLO11n, RT-DETR-L). |
| `experiments/` | Carpeta de salida de cada entrenamiento: `args.yaml`, `results.csv`, curvas, pesos (`best.pt`, `last.pt`). |
| `evaluation/` | Resultados de `model.val(split="test")` y predicciones visuales sobre TEST e imágenes externas. |
| `external_data/` | Imágenes externas sin etiquetas, usadas para pruebas cualitativas de generalización. |
| `documentation/` | Informes, marco teórico, auditorías y evidencia escrita del proyecto. |

Todos los scripts calculan la ruta base del proyecto como `Path(__file__).resolve().parent.parent`, lo que permite ejecutarlos desde cualquier directorio de trabajo sin depender de la ubicación desde la que se invoquen.

### 7.2 Scripts de entrenamiento

Los cuatro modelos se entrenan mediante scripts independientes que siguen el mismo patrón:

1. Cargar el peso base con `YOLO(...)` (o `RTDETR(...)` para RT-DETR-L).
2. Definir rutas absolutas para el dataset, el proyecto y el nombre del experimento.
3. Invocar `model.train(...)` con los hiperparámetros elegidos.

| Script | Modelo | Peso base | Observaciones |
|---|---|---|---|
| `scripts/train_yolov8.py` | YOLOv8n | `models/base/yolov8n.pt` | Entrenamiento local; `data.yaml` apunta históricamente a otra máquina. |
| `scripts/train_yolov10.py` | YOLOv10n | `models/base/yolov10n.pt` | Entrenamiento local; genera un YAML temporal para el dataset. |
| `scripts/train_yolov11.py` | YOLO11n | `models/base/yolo11n.pt` | Diseñado para ejecutarse en Colab; reanuda desde Google Drive. |
| `scripts/train_rtdetr.py` | RT-DETR-L | `models/base/rtdetr-l.pt` | Entrenamiento local; usa `dataset/data.yaml` directamente. |

Cada script respeta la convención del proyecto de escribir los resultados en `experiments/<familia>/<nombre_experimento>/`, con `exist_ok=False` para evitar sobrescribir evidencia previa. La única excepción es YOLO11n, cuyo `args.yaml` indica `exist_ok: true` porque el entrenamiento se reanudó.

### 7.3 Scripts de evaluación y predicción

Además de los scripts de entrenamiento, el repositorio incluye utilidades para evaluar y visualizar el comportamiento de los modelos:

| Script | Función |
|---|---|
| `scripts/test_yolov8.py` | Ejecuta `model.val(split="test")` para YOLOv8n y guarda curvas/batch en `evaluation/yolov8/test_final/`. |
| `scripts/test_yolov10.py` | Evaluación TEST para YOLOv10n en `evaluation/yolov10/test_final/`. |
| `scripts/test_rtdetr.py` | Evaluación TEST para RT-DETR-L en `evaluation/rtdetr/test_final/`. |
| `scripts/predict_test_yolov8.py` | Genera predicciones visuales sobre el split TEST con YOLOv8n. |
| `scripts/predict_test_yolov10.py` | Genera predicciones visuales sobre el split TEST con YOLOv10n. |
| `scripts/predict_test_rtdetr.py` | Genera predicciones visuales sobre el split TEST con RT-DETR-L. |
| `scripts/predict_external_yolov8.py` | Ejecuta inferencia sobre `external_data/images/` con YOLOv8n. |
| `scripts/predict_external_yolov10.py` | Ejecuta inferencia sobre `external_data/images/` con YOLOv10n. |
| `scripts/predict_external_rtdetr.py` | Ejecuta inferencia sobre `external_data/images/` con RT-DETR-L. |
| `scripts/predict_fire_test_yolov8.py` | Predicciones dirigidas a la clase `Fire` sobre TEST con YOLOv8n. |
| `scripts/predict_fire_test_yolov10.py` | Predicciones dirigidas a la clase `Fire` sobre TEST con YOLOv10n. |
| `scripts/dataset_check.py` | Auditoría estructural del dataset (conteos, IDs, coordenadas, imágenes sin etiqueta). |
| `scripts/fix_polygon_labels.py` | Convierte anotaciones poligonales a cajas delimitadoras. |
| `scripts/delete_invalid_labels.py` | Identifica etiquetas vacías o con formato no reconocido. |

**Falta en la rama integrada:** no se encontraron `scripts/test_yolov11.py`, `scripts/predict_test_yolov11.py` ni `scripts/predict_external_yolov11.py`. Las métricas de TEST de YOLO11n provienen de un proceso de evaluación que no quedó registrado como script en el repositorio actual, lo que dificulta reproducirlas directamente.

### 7.4 Integración de los cuatro modelos

En el estado actual del código, **los cuatro modelos operan como detectores independientes**. No existe un script que:

- Cargue simultáneamente los cuatro checkpoints (`best.pt` de cada familia).
- Reciba una imagen de entrada y la procese con los cuatro modelos.
- Aplique un algoritmo de fusión (WBF, NMS sobre la unión de detecciones, votación, etc.).
- Genere una única salida consolidada.

El documento `documentation/MARCO_TEORICO_MODELO_MATEMATICO.md` describe una propuesta de ensamble heterogéneo, pero, como se señala en el Capítulo 6, no se encontró código que la materialice. Por tanto, la "integración" en este informe se refiere únicamente a la comparación de resultados de cuatro entrenamientos paralelos, no a un sistema de detección conjunto en tiempo de ejecución.

### 7.5 Servicio de inferencia y despliegue

No se encontró en el repositorio ninguna implementación de un servicio de inferencia persistente:

- No hay API REST, servidor web, contenedor Docker ni script de despliegue.
- No hay interfaz gráfica ni aplicación móvil.
- La inferencia se realiza de forma **offline** ejecutando los scripts de predicción mencionados en la sección 7.3.

En consecuencia, el despliegue actual está limitado a la ejecución manual de scripts en un entorno con Python, PyTorch, Ultralytics y una GPU compatible. Cualquier aplicación productiva requeriría construir por separado un servicio de inferencia (por ejemplo, con FastAPI, ONNX Runtime o TensorRT) que no está presente en la evidencia actual.

---

## Capítulo 8. Conclusiones, limitaciones y trabajo futuro

### 8.1 Conclusiones generales

SmartRoad AI es un proyecto de detección de objetos orientado a la seguridad vial, con seis clases que combinan vehículos (`Bus`, `Camion`, `Carro`, `Moto`) y eventos de riesgo (`Accidente`, `Fire`). El dataset actual, verificado directamente en el repositorio, contiene 39.742 imágenes de 640 × 640 píxeles y 177.742 anotaciones válidas en formato YOLO de detección, además de 11 anotaciones poligonales no convertidas y cuatro imágenes de entrenamiento sin etiqueta.

Se entrenaron cuatro arquitecturas representativas del estado del arte en detección en tiempo real: YOLOv8n, YOLOv10n, YOLO11n y RT-DETR-L. Los resultados de validación muestran un orden de desempeño claro, encabezado por YOLO11n (mAP@50 = 0,871; mAP@50-95 = 0,622), seguido de RT-DETR-L (0,855; 0,606), YOLOv10n (0,820; 0,540) y YOLOv8n (0,801; 0,545). La única métrica de TEST disponible corresponde a YOLO11n (mAP@50 = 0,8716; mAP@50-95 = 0,6225), valor consistente con su validación.

El análisis por clase sobre TEST revela que `Fire` es la clase más desafiante (mAP@50-95 = 0,3698), mientras que `Accidente` y `Bus` alcanzan los mejores resultados. Esto coincide con la hipótesis metodológica planteada en el Capítulo 3: las clases con formas irregulares o similares a fondos de alto contraste requieren atención específica en futuras iteraciones.

### 8.2 Limitaciones

El proyecto presenta limitaciones que deben tenerse en cuenta al interpretar los resultados:

1. **Falta de estandarización experimental.** Los cuatro modelos no fueron entrenados con el mismo batch size ni el mismo número de épocas; YOLO11n además se entrenó en Colab con `batch=16` y reanudación desde checkpoint.
2. **Versiones del dataset no verificables.** Los `args.yaml` apuntan a rutas distintas (máquina local de Omar, Colab, copia local), por lo que no es posible confirmar binariamente que todos usaron exactamente el mismo conjunto de imágenes y anotaciones.
3. **Métricas de TEST incompletas.** Solo YOLO11n tiene métricas de TEST persistidas. Las carpetas de evaluación de los otros tres modelos contienen curvas y predicciones visuales, pero no un resumen numérico equivalente.
4. **Etiquetas poligonales sin convertir.** Existen 11 anotaciones de `Accidente` en formato poligonal. No se encontró evidencia de que `fix_polygon_labels.py` haya sido ejecutado, por lo que no se sabe si fueron ignoradas o convertidas durante el entrenamiento.
5. **Imágenes sin etiqueta.** Cuatro imágenes de entrenamiento carecen de archivo de etiqueta. Su contenido visual sugiere que podrían contener objetos anotables, por lo que su estatus como ejemplos negativos no está confirmado.
6. **Reproducibilidad ambiental.** No se encontró `requirements.txt`, `environment.yml` ni log de versiones, lo que dificulta replicar el entorno exacto.
7. **Ensamble no implementado.** El modelo matemático de acoplamiento multimodelo propuesto en `MARCO_TEORICO_MODELO_MATEMATICO.md` no tiene correspondencia en código ejecutable.
8. **Sin servicio de inferencia.** No existe API, contenedor ni aplicación para despliegue; la inferencia es exclusivamente offline mediante scripts.

### 8.3 Trabajo futuro

Para robustecer el proyecto y cerrar las brechas identificadas, se propone:

- **Persistir métricas de TEST** para YOLOv8n, YOLOv10n y RT-DETR-L, de preferencia con el mismo script que generó las de YOLO11n.
- **Ejecutar un experimento controlado** en el que los cuatro modelos entrenen desde el mismo `data.yaml` y con hiperparámetros comparables (mismo batch, mismas épocas base, semilla común).
- **Resolver el estado de las anotaciones poligonales:** convertirlas a cajas delimitadoras o confirmar explícitamente que fueron excluidas del entrenamiento.
- **Aclarar las cuatro imágenes sin etiqueta**, determinando si son negativos intencionales u omisiones.
- **Implementar y evaluar el ensamble** propuesto, comparándolo contra cada modelo individual sobre TEST.
- **Crear un servicio de inferencia** (por ejemplo, FastAPI + ONNX/TensorRT) que permita desplegar el modelo en un entorno de producción.
- **Documentar el entorno** mediante un `requirements.txt` o `environment.yml`, y fijar las versiones de Ultralytics, PyTorch y CUDA.
- **Reforzar las clases críticas**, especialmente `Fire`, mediante técnicas de balanceo, aumentación específica o recolección de más ejemplos diversos.

---

## Capítulo 9. Referencias bibliográficas

[1] Fang, J., Qiao, J., Xue, J., & Li, Z. (2023). *Vision-Based Traffic Accident Detection and Anticipation: A Survey*. arXiv:2308.15985. https://arxiv.org/abs/2308.15985

[2] Razi, A., Chen, X., Li, H., Wang, H., & Russo, B. (2023). Deep learning serves traffic safety analysis: A forward-looking review. *IET Intelligent Transport Systems*, 17(4), 618–640. https://doi.org/10.1049/itr2.12257

[3] Zhao, R., Tang, S., Supeni, E. E., Rahim, S. B. A., & Fan, L. (2024). A Review of Object Detection in Traffic Scenes Based on Deep Learning. *Applied Mathematics and Nonlinear Sciences*, 9(1). https://doi.org/10.2478/amns-2024-0322

[4] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 779–788. https://doi.org/10.1109/CVPR.2016.91

[5] Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (Version 8.0.0) [Software]. https://github.com/ultralytics/ultralytics

[6] Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., & Ding, G. (2024). YOLOv10: Real-Time End-to-End Object Detection. *Advances in Neural Information Processing Systems*, 37. arXiv:2405.14458. https://arxiv.org/abs/2405.14458

[7] Jocher, G., & Qiu, J. (2024). *Ultralytics YOLO11* [Software]. https://docs.ultralytics.com/models/yolo11

[8] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end object detection with transformers. *European Conference on Computer Vision (ECCV)*, 213–229. arXiv:2005.12872. https://doi.org/10.1007/978-3-030-58452-8_13

[9] Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J. (2024). DETRs beat YOLOs on real-time object detection. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16965–16974. arXiv:2304.08069. https://doi.org/10.1109/CVPR.2024.01603

[10] Solovyev, R., Wang, W., & Gabruseva, T. (2021). Weighted boxes fusion: Ensembling boxes from different object detection models. *Image and Vision Computing*, 107, 104117. https://doi.org/10.1016/j.imavis.2021.104117

> **Nota editorial:** esta lista se irá completando a medida que se redacten los capítulos restantes. Las entradas se han construido a partir de los metadatos disponibles en arXiv, IEEE Xplore, Crossref, y en la documentación oficial de Ultralytics.

---

## Capítulo 10. Anexos y evidencias complementarias

### Anexo A. Artefactos de entrenamiento

- `experiments/yolov8/yolov8n_20epochs-2/`
- `experiments/yolov10/entrenamiento_01_20_epochs/`
- `experiments/yolov11/entrenamiento_02_50_epochs_colab/`
- `experiments/rtdetr/entrenamiento_01_25_epochs/`

### Anexo B. Artefactos de evaluación

- `evaluation/yolov8/test_final/`
- `evaluation/yolov10/test_final/`
- `evaluation/yolov11/test_final/metrics_summary.txt`
- `evaluation/rtdetr/test_final/`

### Anexo C. Pruebas con imágenes externas

- `external_data/images/`
- `evaluation/*/external_test/results/`

### Anexo D. Scripts y configuraciones

- `scripts/train_*.py`
- `scripts/test_*.py`
- `scripts/predict_*.py`
- `dataset/data.yaml`

### Anexo E. Listado de datos pendientes de verificación

1. Métricas de TEST persistidas para YOLOv8, YOLOv10 y RT-DETR.
2. Scripts de evaluación/predicción para YOLO11 (`test_yolov11.py`, `predict_test_yolov11.py`, `predict_external_yolov11.py`).
3. Coherencia de las pruebas externas de YOLOv8 y YOLOv10 con las 25 imágenes disponibles.
4. Checkpoint real usado en la prueba externa de YOLOv8 (el resumen apunta a una ruta inexistente).
5. Versión exacta del dataset usada por cada modelo (`args.yaml` apunta a rutas distintas).
6. Estado de conversión de las etiquetas poligonales.
7. Estatus de las cuatro imágenes de entrenamiento sin etiqueta (negativos intencionales u omisión).
8. Existencia e implementación de un servicio de inferencia o ensemble.
9. Archivo de dependencias (`requirements.txt` / `environment.yml`).

---

## Control de versiones del documento

| Versión | Fecha | Descripción | Autor |
|---|---|---|---|
| 0.2 | 2026-10-28 | Capítulos 1–8 redactados; correcciones de consistencia en el dataset y estados del arte/modelos/metodología/resultados. | [PENDIENTE] |

---

**Fin de la estructura preliminar.**
