# SmartRoad AI — Auditoría y reorganización del proyecto

Fecha: 2026-09-24
Estado: Fase 2 ejecutada (movimientos realizados con `git mv`, sin eliminar archivos)
Alcance: organización de carpetas para reproducibilidad y documentación académica

---

## 1. Resumen

Se reorganizó el proyecto separando dataset, scripts, modelos base, experimentos de
entrenamiento, evaluación sobre test y documentación. No se eliminó ni sobrescribió
ningún archivo de resultados. Las únicas ediciones de código fueron las rutas
imprescindibles de `scripts/test_yolov8.py` y `scripts/train_yolov8.py` (autorizadas).

El dataset permanece intacto:

- Train: 37.106 imágenes / 37.106 labels
- Valid: 5.305 imágenes / 5.305 labels
- Test: 2.648 imágenes / 2.648 labels
- Clases (`dataset/data.yaml`): `['Accidente', 'Bus', 'Camion', 'Carro', 'Fire', 'Moto']`

## 2. Estructura final

```text
Entrenamiento_Proyecto/
├── dataset/                                       (intacto)
├── scripts/
│   ├── dataset_check.py
│   ├── fix_polygon_labels.py
│   ├── test_yolov8.py                             (ruta de modelo actualizada)
│   └── train_yolov8.py                            (rutas de modelo/proyecto actualizadas)
├── models/
│   └── base/
│       ├── yolov8n.pt
│       └── yolo26n.pt
├── experiments/
│   └── yolov8/
│       ├── sanity_check/
│       ├── entrenamiento_01_10_epochs/
│       ├── entrenamiento_02_20_epochs_definitivo/
│       └── intentos_interrumpidos/
│           ├── yolov8n_10epochs-2/
│           └── yolov8n_10epochs_args/
├── evaluation/
│   └── yolov8/
│       └── test_final/
├── documentation/
│   └── AUDITORIA_Y_REORGANIZACION.md
├── README.md
└── .gitignore
```

## 3. Experimentos

| Experimento | Descripción | Epochs | Métricas finales (validación) |
|---|---|---|---|
| `sanity_check/` | Prueba mínima con modelo base `yolov8n.pt`, batch 2, ~25 min (2026-09-23 00:33–00:59) | 1 | P 0.5175 / R 0.5422 / mAP50 0.5196 / mAP50-95 0.2850 |
| `entrenamiento_01_10_epochs/` | Entrenamiento de 10 epochs desde `yolov8n.pt` (2026-09-23 01:12–04:32) | 10 | P 0.7088 / R 0.6383 / mAP50 0.6908 / mAP50-95 0.4402 |
| `entrenamiento_02_20_epochs_definitivo/` | **Entrenamiento definitivo** de 20 epochs (resume del propio `last.pt`, `save_period=1`, 2026-09-24 01:20–18:01) | 20 | P 0.8028 / R 0.7629 / mAP50 0.8240 / mAP50-95 0.5420 |
| `intentos_interrumpidos/yolov8n_10epochs-2/` | Segundo intento de 10 epochs, interrumpido; sin `results.csv`, `weights/` vacío | — | — |
| `intentos_interrumpidos/yolov8n_10epochs_args/` | `args.yaml` de un intento previo interrumpido antes del epoch 1 | — | — |

### 3.1 Evidencia del entrenamiento definitivo

Contiene 44 archivos: `weights/best.pt`, `weights/last.pt`, `weights/epoch0.pt` … `weights/epoch19.pt`,
`args.yaml`, `results.csv` (20 filas), `results.png`, matrices de confusión, curvas
Box P/R/PR/F1, `labels.jpg`, `train_batch*.jpg`, `val_batch*_{labels,pred}.jpg`.

Hashes SHA-256 de referencia:

| Archivo | SHA-256 |
|---|---|
| `weights/best.pt` (= `last.pt`) | `7009848C056A633350A8314C5F8B3F76248A21232A5BD4B00E42A602B83AE8C3` |
| `weights/epoch0.pt` | `534AC3F5860468CCFCBDA7766618F148797F1028E9BD79358B48EA95B4D71CC5` |
| `weights/epoch19.pt` | `AA9E5079BED2512805996453667C0A4EBC96747BB4D0E190578C53AFDE84C5CA` |

Nota: `best.pt` y `last.pt` son idénticos porque el mejor epoch fue el último.
Los 20 `epoch*.pt` (≈17,9 MB c/u) son checkpoints únicos y se conservan todos.

## 4. Evaluación final sobre TEST

Carpeta: `evaluation/yolov8/test_final/` (12 archivos: 6 PNG de métricas/curvas + 6 JPG de batches).

- Producida por `scripts/test_yolov8.py` con `split="test"` (2026-09-24 20:04).
- Modelo evaluado: `experiments/yolov8/entrenamiento_02_20_epochs_definitivo/weights/best.pt`
  (SHA-256 `7009848C…`).
- Contiene `confusion_matrix.png` (SHA-256 `EF04F073ED58B0A1285DD56EA8323FA331712D816086974B1C836382979EF345`),
  `confusion_matrix_normalized.png` y curvas Box P/R/PR/F1.
- Verificación de que es TEST y no validación: los conteos de la matriz son ~2x menores
  que los del `confusion_matrix.png` de validación del entrenamiento (p. ej. Moto 7.858 vs 13.681),
  consistente con test (2.648) vs valid (5.305).
- Limitación: esta evaluación no generó `results.csv`; las métricas numéricas agregadas
  solo quedaron en la consola y en las gráficas.

## 5. Duplicados verificados (SHA-256)

| Archivo A | Archivo B | SHA-256 | Estado |
|---|---|---|---|
| `models/base/yolov8n.pt` | `scripts/yolov8n.pt` | `F59B3D833E2FF32E194B5BB8E08D211DC7C5BDF144B90D2C8412C47CCFC83B36` | Idénticos. Duplicado eliminado el 2026-09-24 con autorización (se conserva `models/base/yolov8n.pt`) |
| `models/base/yolo26n.pt` | `scripts/weights/yolo26n.pt` | `9B09CC8BF347F0FC8A5F7657480587F25DB09B34BF33B0652110FB03A8AD4FEF` | Idénticos. Duplicado eliminado el 2026-09-24 con autorización (se conserva `models/base/yolo26n.pt`) |
| `entrenamiento_01_10_epochs/weights/best.pt` | `…/last.pt` | `245803E730F2180C4B8B5156723B739ACFB0423B556E8DBD504935CEECDE056C` | Idénticos (no eliminar; evidencia) |
| `sanity_check/weights/best.pt` | `…/last.pt` | `F9021D07B8E0EA82BFD8542F0A29CAA18D45AC6A013BF5AE78B1C9C64929B9F0` | Idénticos (no eliminar; evidencia) |
| `entrenamiento_02_20epochs/weights/best.pt` | `…/last.pt` | `7009848C…` | Idénticos (no eliminar; evidencia) |

También son idénticos 7 recortes de imagen temprana entre `sanity_check/` y
`entrenamiento_01_10_epochs/` (`labels.jpg`, `train_batch0-2.jpg`, `val_batch*_labels.jpg`);
se conservan en ambos por ser evidencia (≈1,5 MB en total).

## 6. Movimientos ejecutados

| Origen | Destino |
|---|---|
| `scripts/runs/runs/smartroad/yolov8n_20epochs/` | `experiments/yolov8/entrenamiento_02_20_epochs_definitivo/` |
| `runs/smartroad/yolov8n_10epochs/` | `experiments/yolov8/entrenamiento_01_10_epochs/` |
| `runs/smartroad/yolov8n_10epochs-2/` | `experiments/yolov8/intentos_interrumpidos/yolov8n_10epochs-2/` |
| `scripts/runs/detect/runs/smartroad/yolov8n_10epochs/` | `experiments/yolov8/intentos_interrumpidos/yolov8n_10epochs_args/` |
| `runs/detect/runs/sanity/sanity_check/` | `experiments/yolov8/sanity_check/` |
| `scripts/runs/detect/val-2/` | `evaluation/yolov8/test_final/` |
| `yolov8n.pt` (raíz) | `models/base/yolov8n.pt` |
| `weights/yolo26n.pt` | `models/base/yolo26n.pt` |

Método: `git mv` para contenido rastreado (preserva historial) y `Move-Item` para
`val-2/` (no rastreado). Verificación posterior: hashes de `best.pt`/`last.pt`/`epoch*.pt`
sin cambios, 44 archivos en el experimento definitivo y 12 en `test_final/`.

## 7. Cambios en scripts (solo rutas)

`scripts/test_yolov8.py`:

```python
MODEL_PATH = PROJECT_DIR / "experiments" / "yolov8" / "entrenamiento_02_20_epochs_definitivo" / "weights" / "best.pt"
```

`scripts/train_yolov8.py`:

```python
model = YOLO("../models/base/yolov8n.pt")
project="../experiments/yolov8",
```

No se modificó la lógica de entrenamiento/evaluación. Los `args.yaml` de cada run
conservan sus rutas históricas originales (son evidencia y no se editan).

## 8. Limpieza ejecutada y pendientes

Limpieza ejecutada el 2026-09-24 con autorización explícita, tras re-verificar hashes
idénticos y confirmar que ningún script los referenciaba:

1. `scripts/yolov8n.pt` — eliminado (`git rm`); duplicado exacto de `models/base/yolov8n.pt`.
2. `scripts/weights/yolo26n.pt` — eliminado (`git rm`); duplicado exacto de `models/base/yolo26n.pt`.
3. `scripts/runs/detect/val/` — eliminado; carpeta vacía.
4. Directorios vacíos restantes de los movimientos — eliminados: `runs/` (árbol completo),
   `weights/`, `scripts/runs/` (árbol completo), `scripts/weights/` y los `weights/` vacíos de
   `intentos_interrumpidos/`.

Ninguna evidencia de experimentos fue eliminada: los conteos se mantienen (44 archivos en el
entrenamiento definitivo, 21 en 10epochs, 21 en sanity, 12 en test_final, 5 y 1 en los intentos).

Pendientes (requieren autorización):

1. Reescribir `README.md` (actualmente es un placeholder UTF-16 de 64 bytes:
   `# Entrenamiento_DataScience`).
2. Opcional: eliminar `.idea/` (configuración local de IDE, ya ignorada por git).
3. Documentar las métricas numéricas de la evaluación TEST (hoy solo en gráficas/consola).

## 9. Reproducción

- Entrenamiento: ejecutar `scripts/train_yolov8.py` (20 epochs, base `models/base/yolov8n.pt`,
  salida en `experiments/yolov8/`).
- Evaluación TEST: ejecutar `scripts/test_yolov8.py` (modelo
  `experiments/yolov8/entrenamiento_02_20_epochs_definitivo/weights/best.pt`, `split="test"`).
- Dataset: `dataset/data.yaml` (train 37.106 / valid 5.305 / test 2.648).
