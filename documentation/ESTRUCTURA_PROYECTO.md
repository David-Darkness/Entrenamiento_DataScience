# SmartRoad AI — Estructura del proyecto

Fecha: 2026-09-24
Propósito: referencia oficial de arquitectura para que cualquier script (humano o IA)
guarde sus salidas en el lugar correcto y no se vuelva a desordenar el proyecto.
Documento relacionado: `documentation/AUDITORIA_Y_REORGANIZACION.md` y `AGENTS.md` (raíz).

---

## 1. Árbol completo

```text
Entrenamiento_Proyecto/
│
├── dataset/                                      # INTACTO: no mover, renombrar ni editar
│   ├── data.yaml                                 # nc=6 → Accidente, Bus, Camion, Carro, Fire, Moto
│   ├── train/  { images/ (37.106), labels/ (37.106) }
│   ├── valid/  { images/ (5.305),  labels/ (5.305) }
│   ├── test/   { images/ (2.648),  labels/ (2.648) }
│   └── labels_backup/ { train/labels, test/labels }
│
├── scripts/                                      # SOLO código .py — ninguna salida aquí
│   ├── train_yolov8.py                           # entrenamiento (salida en ../experiments/yolov8)
│   ├── test_yolov8.py                            # evaluación split=test (salida en ../evaluation/yolov8)
│   ├── predict_test_yolov8.py                    # predicciones visuales aleatorias de TEST
│   ├── predict_fire_test_yolov8.py               # prueba dirigida de la clase Fire en TEST
│   ├── dataset_check.py
│   └── fix_polygon_labels.py
│
├── models/
│   └── base/                                     # ÚNICO lugar para pesos preentrenados
│       ├── yolov8n.pt
│       └── yolo26n.pt
│
├── experiments/                                  # una subcarpeta por FAMILIA de modelo
│   └── yolov8/
│       ├── sanity_check/                         # 1 epoch, prueba mínima (21 archivos)
│       │   ├── args.yaml
│       │   ├── results.csv | results.png
│       │   ├── confusion_matrix.png | confusion_matrix_normalized.png
│       │   ├── BoxP_curve.png | BoxR_curve.png | BoxPR_curve.png | BoxF1_curve.png
│       │   ├── labels.jpg | train_batch*.jpg | val_batch*_{labels,pred}.jpg
│       │   └── weights/ { best.pt, last.pt }
│       ├── entrenamiento_01_10_epochs/           # 10 epochs (21 archivos, misma estructura)
│       ├── entrenamiento_02_20_epochs_definitivo/
│       │   ├── (misma estructura + train_batch185530-32.jpg)
│       │   └── weights/ { best.pt, last.pt, epoch0.pt … epoch19.pt }   # 44 archivos
│       └── intentos_interrumpidos/
│           ├── yolov8n_10epochs-2/               # args.yaml + labels/batches (sin weights)
│           └── yolov8n_10epochs_args/            # solo args.yaml
│
├── evaluation/                                   # una subcarpeta por FAMILIA de modelo
│   └── yolov8/
│       └── test_final/                           # evaluación split=test del definitivo
│           ├── confusion_matrix.png | confusion_matrix_normalized.png
│           ├── BoxP_curve.png | BoxR_curve.png | BoxPR_curve.png | BoxF1_curve.png
│           ├── val_batch*_{labels,pred}.jpg
│           └── predictions/
│               ├── muestras_generales/           # predicciones aleatorias sobre TEST
│               └── fire_test/results/            # prueba dirigida de Fire sobre TEST
│
├── documentation/
│   ├── AUDITORIA_Y_REORGANIZACION.md
│   └── ESTRUCTURA_PROYECTO.md                    # este documento
│
├── AGENTS.md                                     # reglas para asistentes/IA
├── README.md
├── .gitignore
├── .idea/                                        # config IDE local (ignorada por git)
└── .git/
```

## 2. Dónde va cada cosa

| Qué genera el script | Dónde va | Cómo forzarlo |
|---|---|---|
| Peso base preentrenado (`.pt`) | `models/base/` | ruta fija |
| Entrenamiento | `experiments/<familia>/entrenamiento_XX_<descripcion>/` | `project=` + `name=` en `model.train()` |
| Evaluación TEST | `evaluation/<familia>/<nombre>/` | `project=` + `name=` en `model.val()` |
| Predicciones visuales | `evaluation/<familia>/<nombre>/predictions/<grupo>/` | `project=` + `name=` si usa `save=True` |
| Documentación | `documentation/` | — |
| Scripts | `scripts/` | — |

Convención de nombres:

- Familia de modelo: `yolov8`, `yolo26`, …
- Experimento: `entrenamiento_XX_<descripcion>` (XX = consecutivo, descripción corta).
- Evaluación: `test_final` (o `test_final2`, etc. si ya existe; nunca sobrescribir).
- Grupo de predicciones: `muestras_generales`, `fire_test`, `<clase>_test`, …

## 3. Reglas obligatorias

1. `dataset/` es intocable: solo lectura (imágenes, labels, `data.yaml`, `labels_backup/`).
2. Nunca crear `runs/` (ni en la raíz ni en `scripts/`) y nunca escribir salidas en `scripts/`.
3. Pesos base preentrenados solo en `models/base/`.
4. Todo entrenamiento debe fijar `project` y `name` en `model.train()`, con `name` único
   (si se repite, Ultralytics crea sufijos `-2`, `-3`… y desordena).
5. Toda evaluación debe fijar `project` y `name` en `model.val()`. Sin ellos, Ultralytics
   escribe en `runs/detect/val*` fuera de `evaluation/`.
6. Usar rutas absolutas desde `PROJECT_DIR = Path(__file__).resolve().parent.parent`
   (funciona desde cualquier directorio de ejecución).
7. No sobrescribir evidencia: `exist_ok=False` y nombres nuevos. Si el nombre ya existe,
   Ultralytics crea un sufijo numérico en lugar de sobrescribir.
8. Nunca eliminar `best.pt`, `last.pt`, `epoch*.pt`, `args.yaml`, `results.csv`,
   `results.png`, matrices de confusión, curvas, `labels.jpg`, `train_batch*`, `val_batch*`
   ni predicciones de `evaluation/`.

## 4. Plantilla para el modelo 2

```python
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent

model = YOLO(str(PROJECT_DIR / "models" / "base" / "yolo26n.pt"))

model.train(
    data=str(PROJECT_DIR / "dataset" / "data.yaml"),
    project=str(PROJECT_DIR / "experiments" / "yolo26"),   # familia del modelo 2
    name="entrenamiento_01_<descripcion>",                 # unico, nunca repetido
    epochs=...,
    imgsz=640,
    batch=2,
    device=0,
    workers=0,
    save=True,
    save_period=1,
    plots=True,
    seed=0,
    patience=...,
)
```

Evaluación sobre TEST del modelo 2:

```python
model.val(
    data=str(PROJECT_DIR / "dataset" / "data.yaml"),
    split="test",
    project=str(PROJECT_DIR / "evaluation" / "yolo26"),
    name="test_final",
    imgsz=640,
    batch=2,
    device=0,
    workers=0,
    plots=True,
)
```

## 5. Estado actual

- Modelo 1 (YOLOv8n) definitivo: `experiments/yolov8/entrenamiento_02_20_epochs_definitivo/weights/best.pt`
  (mAP50 0.824 / mAP50-95 0.542). SHA-256 de `best.pt`:
  `7009848C056A633350A8314C5F8B3F76248A21232A5BD4B00E42A602B83AE8C3`.
- Evaluación TEST del modelo 1: `evaluation/yolov8/test_final/`.
- Predicciones: `evaluation/yolov8/test_final/predictions/`.
- Modelo 2: por entrenar. Crear `experiments/<familia_modelo2>/` para el entrenamiento y
  `evaluation/<familia_modelo2>/test_final/` para la evaluación.
