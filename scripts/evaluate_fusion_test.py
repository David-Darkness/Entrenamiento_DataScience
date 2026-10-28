"""Evaluación sobre TEST de los cuatro modelos y de la fusión ponderada de la app.

Ejecuta YOLOv8n, YOLOv10n, YOLO11n y RT-DETR-L sobre dataset/test/images, guarda
las predicciones crudas (caché JSON) y compara, con un mismo evaluador:

  - cada modelo individual (puntuación = confianza del modelo), y
  - la fusión ponderada del motor de decisión de la aplicación
    (puntuación = ŝ = s̄·Φ, Capítulo 6 del informe técnico).

Métricas: AP@50 por clase, mAP@50 y, para un barrido de umbrales, precisión,
recall y F1 (IoU >= 0,5, emparejamiento voraz por puntuación). Además reporta el
umbral mínimo con el que cada método alcanza una precisión >= 0,95.

Uso (desde la raíz del proyecto):
    python scripts/evaluate_fusion_test.py                 # evaluación completa
    python scripts/evaluate_fusion_test.py --max-images 50 # prueba rápida
    python scripts/evaluate_fusion_test.py --reuse         # reutiliza la caché

Salidas (no sobrescribe otras evaluaciones):
    evaluation/fusion/test_final/
        predicciones_<modelo>.json, metrics_summary.txt,
        metricas_por_metodo.csv, ap50_por_clase.csv, barrido_umbral.csv,
        precision_recall_umbral.png

Nota: el AP de este evaluador (101 puntos, emparejamiento propio) puede diferir
ligeramente del que reporta Ultralytics en model.val(); las comparaciones válidas
son las que se hacen dentro de este mismo evaluador.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from app_modules.decision_engine import FusionConfig, fuse_detections  # noqa: E402
from app_modules.inference import Detection  # noqa: E402
from app_modules.model_manager import EXPECTED_CLASSES, build_registry, load_models  # noqa: E402

CLASSES = [EXPECTED_CLASSES[i] for i in sorted(EXPECTED_CLASSES)]
IOU_MATCH = 0.5
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
OUT_DIR = PROJECT_ROOT / "evaluation" / "fusion" / "test_final"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def load_ground_truth(label_path: Path, width: int, height: int) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """Lee etiquetas YOLO (cajas o polígonos) y las devuelve en píxeles xyxy."""
    gts = []
    if not label_path.exists():
        return gts
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            xc, yc, w, h = vals
            box = ((xc - w / 2) * width, (yc - h / 2) * height, (xc + w / 2) * width, (yc + h / 2) * height)
        else:  # polígono -> rectángulo envolvente
            xs, ys = vals[0::2], vals[1::2]
            box = (min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height)
        gts.append((EXPECTED_CLASSES.get(cls, str(cls)), box))
    return gts


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Inferencia con caché
# ---------------------------------------------------------------------------
def run_models(images: Sequence[Path], device: str, conf: float, imgsz: int, reuse: bool) -> Dict[str, Dict[str, list]]:
    """Devuelve {modelo: {imagen: [[cls, conf, x1, y1, x2, y2], ...]}}."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = build_registry(PROJECT_ROOT)
    cached: Dict[str, Dict[str, list]] = {}
    pending = {}
    for key, info in registry.items():
        cache_file = OUT_DIR / f"predicciones_{key}.json"
        if reuse and cache_file.exists():
            data = json.loads(cache_file.read_text())
            if data.get("conf") <= conf and all(p.name in data["preds"] for p in images):
                cached[info.display_name] = data["preds"]
                print(f"[caché] {info.display_name}: {cache_file.name}")
                continue
        pending[key] = info

    if pending:
        models = load_models({k: registry[k] for k in pending}, device=device)
        for key, res in models.items():
            name = res.info.display_name
            if not res.available:
                raise RuntimeError(f"No se pudo cargar {name}: {res.error}")
            preds: Dict[str, list] = {}
            start = time.time()
            for i, img in enumerate(images, 1):
                r = res.model(str(img), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
                b = r.boxes
                rows = []
                if b is not None and len(b):
                    for c, s, xyxy in zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xyxy.cpu().numpy()):
                        rows.append([int(c), float(s), *[float(v) for v in xyxy]])
                preds[img.name] = rows
                if i % 200 == 0 or i == len(images):
                    print(f"  {name}: {i}/{len(images)} imágenes ({time.time() - start:.0f} s)")
            cached[name] = preds
            (OUT_DIR / f"predicciones_{key}.json").write_text(
                json.dumps({"model": name, "conf": conf, "imgsz": imgsz, "preds": preds})
            )
    return cached


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def to_detections(rows: list, min_conf: float) -> List[Detection]:
    return [
        Detection(class_id=r[0], class_name=EXPECTED_CLASSES[r[0]], confidence=r[1], x1=r[2], y1=r[3], x2=r[4], y2=r[5])
        for r in rows
        if r[1] >= min_conf
    ]


def match_image(preds: List[Tuple[str, float, tuple]], gts: List[Tuple[str, tuple]]) -> List[Tuple[str, float, bool]]:
    """Emparejamiento voraz por puntuación: devuelve (clase, puntuación, es_TP)."""
    out = []
    used = set()
    for cls, score, box in sorted(preds, key=lambda p: -p[1]):
        best, best_j = IOU_MATCH, -1
        for j, (gcls, gbox) in enumerate(gts):
            if gcls != cls or j in used:
                continue
            v = iou(box, gbox)
            if v >= best:
                best, best_j = v, j
        if best_j >= 0:
            used.add(best_j)
        out.append((cls, score, best_j >= 0))
    return out


def average_precision(scores: np.ndarray, tps: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = np.cumsum(tps[order])
    fp = np.cumsum(~tps[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # NumPy 2.x / 1.x
    return float(integrate(np.interp(x, mrec, mpre), x))


def evaluate(method_preds: Dict[str, List[Tuple[str, float, tuple]]], gt: Dict[str, list]) -> dict:
    records = []  # (clase, puntuación, tp)
    n_gt = defaultdict(int)
    for name, gts in gt.items():
        for gcls, _ in gts:
            n_gt[gcls] += 1
        records.extend(match_image(method_preds.get(name, []), gts))
    cls_arr = np.array([r[0] for r in records])
    sc = np.array([r[1] for r in records], dtype=float)
    tp = np.array([r[2] for r in records], dtype=bool)
    ap = {c: average_precision(sc[cls_arr == c], tp[cls_arr == c], n_gt[c]) for c in CLASSES}
    total_gt = sum(n_gt.values())
    sweep = []
    for t in THRESHOLDS:
        m = sc >= t
        tps, fps = int(tp[m].sum()), int((~tp[m]).sum())
        p = tps / (tps + fps) if tps + fps else float("nan")
        r = tps / total_gt if total_gt else float("nan")
        f1 = 2 * p * r / (p + r) if p == p and r == r and p + r > 0 else float("nan")
        sweep.append({"umbral": t, "precision": p, "recall": r, "f1": f1, "TP": tps, "FP": fps})
    valid = [v for v in ap.values() if v == v]
    return {"ap50": ap, "map50": float(np.mean(valid)) if valid else float("nan"), "sweep": sweep, "n_gt": dict(n_gt)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None, help="'0' para GPU o 'cpu' (por defecto: GPU si existe)")
    ap.add_argument("--conf", type=float, default=0.05, help="confianza mínima guardada en la caché")
    ap.add_argument("--app-conf", type=float, default=0.25, help="umbral de confianza de la app antes de fusionar")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-images", type=int, default=0, help="0 = todas")
    ap.add_argument("--reuse", action="store_true", help="reutilizar predicciones en caché")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--iou-fusion", type=float, default=0.5)
    args = ap.parse_args()

    import torch

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    img_dir = PROJECT_ROOT / "dataset" / "test" / "images"
    lbl_dir = PROJECT_ROOT / "dataset" / "test" / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.max_images:
        images = images[: args.max_images]
    print(f"Imágenes de TEST: {len(images)} | dispositivo: {device}")

    gt = {}
    for p in images:
        with Image.open(p) as im:
            w, h = im.size
        gt[p.name] = load_ground_truth(lbl_dir / (p.stem + ".txt"), w, h)

    preds = run_models(images, device, args.conf, args.imgsz, args.reuse)

    methods: Dict[str, Dict[str, list]] = {}
    for model_name, per_img in preds.items():
        methods[model_name] = {
            n: [(EXPECTED_CLASSES[r[0]], r[1], tuple(r[2:])) for r in rows if r[1] >= args.conf]
            for n, rows in per_img.items()
        }
    cfg = FusionConfig(iou_threshold=args.iou_fusion, gamma=args.gamma, beta=args.beta)
    fused: Dict[str, list] = {}
    for p in images:
        per_model = {m: to_detections(preds[m][p.name], args.app_conf) for m in preds}
        clusters, weights, leader = fuse_detections(per_model, cfg)
        fused[p.name] = [(c.class_name, c.fused_score, c.fused_box) for cl in clusters.values() for c in cl]
    methods["Fusión (app)"] = fused

    results = {m: evaluate(mp, gt) for m, mp in methods.items()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    order = list(methods)
    with open(OUT_DIR / "ap50_por_clase.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Clase", *order])
        for c in CLASSES:
            w.writerow([c, *[f"{results[m]['ap50'][c]:.4f}" for m in order]])
        w.writerow(["mAP@50", *[f"{results[m]['map50']:.4f}" for m in order]])
    with open(OUT_DIR / "barrido_umbral.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metodo", "Umbral", "Precision", "Recall", "F1", "TP", "FP"])
        for m in order:
            for row in results[m]["sweep"]:
                w.writerow([m, row["umbral"], f"{row['precision']:.4f}", f"{row['recall']:.4f}", f"{row['f1']:.4f}", row["TP"], row["FP"]])

    def best_f1(m):
        return max((r for r in results[m]["sweep"] if r["f1"] == r["f1"]), key=lambda r: r["f1"])

    def first_p95(m):
        return next((r for r in results[m]["sweep"] if r["precision"] == r["precision"] and r["precision"] >= 0.95), None)

    def at(m, t):
        return next(r for r in results[m]["sweep"] if abs(r["umbral"] - t) < 1e-9)

    with open(OUT_DIR / "metricas_por_metodo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metodo", "mAP@50", "P@0.25", "R@0.25", "Umbral F1 max", "P (F1 max)", "R (F1 max)", "F1 max",
                    "Umbral P>=0.95", "P (P>=0.95)", "R (P>=0.95)"])
        for m in order:
            b, q, r25 = best_f1(m), first_p95(m), at(m, 0.25)
            w.writerow([m, f"{results[m]['map50']:.4f}", f"{r25['precision']:.4f}", f"{r25['recall']:.4f}",
                        b["umbral"], f"{b['precision']:.4f}", f"{b['recall']:.4f}", f"{b['f1']:.4f}",
                        q["umbral"] if q else "-", f"{q['precision']:.4f}" if q else "-", f"{q['recall']:.4f}" if q else "-"])

    lines = [
        "SMARTROAD AI - Evaluacion TEST: modelos individuales vs fusion ponderada (app)",
        f"Imagenes: {len(images)} | objetos GT: {sum(results[order[0]]['n_gt'].values())} | IoU de acierto: {IOU_MATCH}",
        f"Fusion: gamma={args.gamma}, beta={args.beta}, Omega_IoU={args.iou_fusion}, conf previa={args.app_conf}, "
        f"lider={leader}, pesos={ {k: round(v, 3) for k, v in weights.items()} }",
        "",
        f"{'Metodo':<14}{'mAP@50':>8}{'P@0.25':>8}{'R@0.25':>8}{'F1max':>8}{'u(F1)':>7}{'u(P>=.95)':>11}{'R(P>=.95)':>11}",
    ]
    for m in order:
        b, q, r25 = best_f1(m), first_p95(m), at(m, 0.25)
        lines.append(f"{m:<14}{results[m]['map50']:>8.4f}{r25['precision']:>8.4f}{r25['recall']:>8.4f}{b['f1']:>8.4f}"
                     f"{b['umbral']:>7.2f}{(q['umbral'] if q else float('nan')):>11.2f}{(q['recall'] if q else float('nan')):>11.4f}")
    lines += ["", "AP@50 por clase:", f"{'Clase':<11}" + "".join(f"{m:>14}" for m in order)]
    for c in CLASSES:
        lines.append(f"{c:<11}" + "".join(f"{results[m]['ap50'][c]:>14.4f}" for m in order))
    text = "\n".join(lines)
    (OUT_DIR / "metrics_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for m in order:
            t = [r["umbral"] for r in results[m]["sweep"]]
            lw = 3 if m.startswith("Fus") else 1.5
            ax[0].plot(t, [r["precision"] for r in results[m]["sweep"]], lw=lw, label=m)
            ax[1].plot(t, [r["recall"] for r in results[m]["sweep"]], lw=lw, label=m)
        ax[0].axhline(0.95, color="grey", ls="--", lw=1)
        ax[0].set_title("Precisión vs umbral (TEST)")
        ax[1].set_title("Recall vs umbral (TEST)")
        for a in ax:
            a.set_xlabel("umbral sobre la puntuación (conf. o ŝ)")
            a.grid(alpha=0.3)
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "precision_recall_umbral.png", dpi=160)
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo generar la gráfica: {exc}")


if __name__ == "__main__":
    main()
