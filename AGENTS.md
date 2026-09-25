# AGENTS.md — SmartRoad AI (Entrenamiento_Proyecto)

Guía obligatoria de estructura para cualquier IA/agente que genere scripts o archivos en este repositorio.

## Contexto

- Proyecto académico de detección de objetos con YOLOv8 sobre el dataset SmartRoad AI.
- Clases (6): `Accidente`, `Bus`, `Camion`, `Carro`, `Fire`, `Moto`.
- Dataset: train 37.106 / valid 5.305 / test 2.648 imágenes.
- Los resultados de entrenamientos y evaluaciones son **EVIDENCIA ACADÉMICA**: nunca eliminarlos, moverlos, renombrarlos ni sobrescribirlos sin autorización explícita.

## Estructura del proyecto (respetarla siempre)

```text
Entrenamiento_Proyecto/
├── dataset/                                      # INTACTO (solo lectura)
│   ├── data.yaml
│   ├── train/{images,labels}/
│   ├── valid/{images,labels}/
│   ├── test/{images,labels}/
│   └── labels_backup/
├── scripts/                                      # SOLO código .py; ninguna salida aquí
│   ├── train_yolov8.py
│   ├── test_yolov8.py
│   ├── predict_test_yolov8.py
│   ├── predict_fire_test_yolov8.py
│   ├── dataset_check.py
│   └── fix_polygon_labels.py
├── models/
│   └── base/                                     # ÚNICO lugar para pesos preentrenados
│       ├── yolov8n.pt
│       └── yolo26n.pt
├── experiments/                                  # una carpeta por familia de modelo
│   └── yolov8/
│       ├── sanity_check/
│       ├── entrenamiento_01_10_epochs/
│       ├── entrenamiento_02_20_epochs_definitivo/
│       └── intentos_interrumpidos/
├── evaluation/                                   # una carpeta por familia de modelo
│   └── yolov8/
│       └── test_final/
│           └── predictions/
├── documentation/
├── README.md
└── .gitignore
```

Cada experimento dentro de `experiments/<familia>/<nombre>/` contiene lo generado por
Ultralytics (`args.yaml`, `results.csv`, `results.png`, `confusion_matrix*.png`,
`Box*_curve.png`, `labels.jpg`, `train_batch*`, `val_batch*`, `weights/`).

## Dónde va cada cosa

| Qué genera el script | Dónde va | Cómo forzarlo |
|---|---|---|
| Peso base preentrenado | `models/base/` | ruta fija |
| Entrenamiento | `experiments/<familia>/entrenamiento_XX_<descripcion>/` | `project=` + `name=` en `model.train()` |
| Evaluación TEST | `evaluation/<familia>/<nombre>/` | `project=` + `name=` en `model.val()` |
| Predicciones visuales | `evaluation/<familia>/<nombre>/predictions/<grupo>/` | `project=` + `name=` si usa `save=True` |
| Documentación | `documentation/` | — |
| Scripts | `scripts/` | — |

## Reglas obligatorias

1. `dataset/` es intocable: imágenes, labels, `data.yaml` y `labels_backup/` solo se leen.
2. Nunca crear `runs/` en la raíz ni en `scripts/`, y nunca escribir salidas dentro de `scripts/`.
3. Los pesos base preentrenados van SOLO en `models/base/`.
4. Todo entrenamiento debe especificar `project` (carpeta de la familia) y `name` (nombre del experimento) en `model.train()`.
5. El `name` de cada experimento debe ser único. Nunca reutilizarlo: Ultralytics crea sufijos `-2`, `-3` y desordena el proyecto.
6. Toda evaluación debe especificar `project` y `name` en `model.val()`. Sin ellos, Ultralytics escribe en `runs/detect/val*`.
7. Usar rutas absolutas derivadas de `PROJECT_DIR = Path(__file__).resolve().parent.parent` para que los scripts funcionen desde cualquier directorio de ejecución.
8. Nunca sobrescribir evidencia existente: usar nombres nuevos y `exist_ok=False`.
9. No modificar scripts existentes sin autorización; los scripts nuevos se ubican en `scripts/`.

## Evidencia que NO se toca

`best.pt`, `last.pt`, `epoch*.pt`, `args.yaml`, `results.csv`, `results.png`,
`confusion_matrix*.png`, `Box{P,R,PR,F1}_curve.png`, `labels.jpg`, `train_batch*`,
`val_batch*` y las predicciones en `evaluation/`.

## Plantillas

### Entrenamiento (modelo nuevo)

```python
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent

model = YOLO(str(PROJECT_DIR / "models" / "base" / "<peso_base>.pt"))

model.train(
    data=str(PROJECT_DIR / "dataset" / "data.yaml"),
    project=str(PROJECT_DIR / "experiments" / "<familia_modelo>"),
    name="entrenamiento_XX_<descripcion>",   # unico, nunca repetido
    ...
)
```

### Evaluación sobre TEST

```python
model.val(
    data=str(PROJECT_DIR / "dataset" / "data.yaml"),
    split="test",
    project=str(PROJECT_DIR / "evaluation" / "<familia_modelo>"),
    name="<nombre_evaluacion>",              # unico; exist_ok=False
    ...
)
```

## Estado actual relevante

- Modelo 1 (YOLOv8n) definitivo: `experiments/yolov8/entrenamiento_02_20_epochs_definitivo/weights/best.pt` (mAP50 0.824 / mAP50-95 0.542).
- Evaluación TEST del modelo 1: `evaluation/yolov8/test_final/`.
- Modelo 2 (próximo): crear `experiments/<familia_modelo2>/` y, tras entrenar, `evaluation/<familia_modelo2>/test_final/`.
