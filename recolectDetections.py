import json
from collections import defaultdict
import numpy as np
from datetime import datetime

# --- Almacén global de detecciones ---
DETECTIONS = []

def log_detection(tracker_id, cls_name, speed_kmh, frame_idx, fps):
    """
    Guarda cada detección para análisis estadístico.
    """
    DETECTIONS.append({
        "tracker_id": int(tracker_id),
        "class": cls_name,
        "speed_kmh": float(speed_kmh),
        "frame": int(frame_idx),
        "time_utc": datetime.utcnow().isoformat()
    })


def generate_statistics(speed_limit=90, bucket_seconds=300, fps=30, video_source="stream"):
    """
    Procesa las detecciones y genera un JSON con estadísticos.
    """
    if not DETECTIONS:
        return {}

    speeds = [d["speed_kmh"] for d in DETECTIONS]
    classes = [d["class"] for d in DETECTIONS]
    ids = set([d["tracker_id"] for d in DETECTIONS])

    # --- Resumen global ---
    summary = {
        "vehicles_total": len(DETECTIONS),
        "unique_ids": len(ids),
        "by_class_counts": {cls: classes.count(cls) for cls in set(classes)},
        "speed_kmh": {
            "mean": float(np.mean(speeds)),
            "min": float(np.min(speeds)),
            "max": float(np.max(speeds)),
            "std": float(np.std(speeds)),
            "percentiles": {
                "p50": float(np.percentile(speeds, 50)),
                "p85": float(np.percentile(speeds, 85)),
                "p95": float(np.percentile(speeds, 95)),
                "p99": float(np.percentile(speeds, 99))
            }
        }
    }

    # --- Distribución de velocidades (bins) ---
    bins = [(0,30),(31,60),(61,90),(91,120),(121,150),(151,9999)]
    bin_counts = []
    for bmin, bmax in bins:
        cnt = sum(1 for s in speeds if bmin <= s <= bmax)
        label = f"{bmin}-{bmax}" if bmax < 9999 else f">{bmin}"
        bin_counts.append({"range": label, "count": cnt})

    # --- Violaciones ---
    violators = [d for d in DETECTIONS if d["speed_kmh"] > speed_limit]
    violations = {
        "speed_limit_kmh": speed_limit,
        "total_violators": len(violators),
        "rate": len(violators)/len(DETECTIONS),
        "top_violators": sorted(
            violators, key=lambda x: x["speed_kmh"], reverse=True
        )[:5]  # top 5
    }

    # --- JSON final ---
    stats = {
        "meta": {
            "video_source": video_source,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "fps": fps
        },
        "summary": summary,
        "speed_distribution": {
            "bins_kmh": bin_counts
        },
        "violations": violations
    }

    return stats


def save_statistics_json(filepath="traffic_stats.json"):
    """
    Guarda los estadísticos en un archivo JSON.
    """
    stats = generate_statistics()
    if stats:
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] Estadísticos guardados en {filepath}")
