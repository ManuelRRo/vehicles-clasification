import argparse
import csv
import time
from collections import defaultdict, deque
from pathlib import Path
import math

import numpy as np
import cv2
from ultralytics import YOLO


# ---------------- Utilidades geométricas ----------------
def point_line_side(px, py, x1, y1, x2, y2):
    """Signo del punto respecto a la línea dirigida A(x1,y1)->B(x2,y2)."""
    v = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    return 1 if v > 0 else (-1 if v < 0 else 0)


def compute_speed_along(history, now, window_s, axis_vec_unit):
    """
    Calcula la velocidad proyectada a lo largo de axis_vec_unit (2D),
    usando puntos en 'history' de la forma (t, X, Y).
    """
    if len(history) < 2:
        return 0.0, None, None

    # Filtramos puntos dentro de la ventana [now - window_s, now]
    t_min = now - window_s
    pts = [(t, x, y) for (t, x, y) in history if t >= t_min]
    if len(pts) < 2:
        # Usamos todo el historial si la ventana es muy reciente
        pts = history

    t0, x0, y0 = pts[0]
    t1, x1, y1 = pts[-1]
    dt = t1 - t0
    if dt <= 0:
        return 0.0, None, None

    # Desplazamiento vectorial
    dx = x1 - x0
    dy = y1 - y0

    # Proyección sobre el eje (para saber si va "hacia adelante" o "atrás")
    proj = dx * axis_vec_unit[0] + dy * axis_vec_unit[1]
    v = proj / dt  # m/s o px/s

    return v, (x0, y0), (x1, y1)


def draw_text(img, text, org, color=(255, 255, 255), scale=0.5, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def parse_polygon(s):
    """Convierte 'x1,y1,x2,y2,...' en lista de puntos [(x1,y1), (x2,y2), ...]."""
    parts = s.split(",")
    if len(parts) % 2 != 0:
        raise ValueError("La cadena de ROI debe tener un número par de valores x,y.")
    coords = list(map(int, parts))
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]


def parse_quad(s):
    """Convierte 'x1,y1,x2,y2,x3,y3,x4,y4' en lista de 4 puntos."""
    pts = parse_polygon(s)
    if len(pts) != 4:
        raise ValueError("Se esperaban exactamente 4 puntos (x1,y1,...,x4,y4).")
    return pts


# ---------------- Argumentos ----------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Aforo + velocidad con YOLO + ByteTrack, con homografía opcional."
    )

    # Modelo / fuente
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo YOLO (.pt).")
    parser.add_argument("--source", type=str, required=True, help="Ruta al video o stream (o 0 para webcam).")
    parser.add_argument("--conf", type=float, default=0.3, help="Confianza mínima de detección.")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Archivo de configuración del tracker.")

    # Línea de conteo
    parser.add_argument(
        "--line",
        type=int,
        nargs=4,
        metavar=("x1", "y1", "x2", "y2"),
        help="Línea de conteo en píxeles (si no se usa homografía para generarla).",
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="A->B",
        choices=["A->B", "B->A"],
        help="Sentido de aproximación para el radar (en qué dirección se considera la infracción).",
    )

    # ROI
    parser.add_argument(
        "--roi",
        type=str,
        help="Polígono de interés en formato 'x1,y1,x2,y2,...'. Solo se cuentan vehículos dentro del ROI.",
    )

    # Filtro de clases
    parser.add_argument(
        "--classes-include",
        type=str,
        nargs="+",
        default=None,
        help="Lista de nombres de clases que queremos incluir (e.g. car truck bus).",
    )

    # Homografía / calibración
    parser.add_argument(
        "--road-quad",
        type=str,
        help=(
            "Cuadrilátero de la carretera en la imagen 'x1,y1,x2,y2,x3,y3,x4,y4' en orden (cerca-izq, cerca-der, "
            "lejos-der, lejos-izq). Si se da, se asume un rectángulo en el mundo."
        ),
    )
    parser.add_argument(
        "--road-width-m",
        type=float,
        default=None,
        help="Ancho de la calzada (en metros) asociado a road-quad.",
    )
    parser.add_argument(
        "--road-length-m",
        type=float,
        default=None,
        help="Largo de la calzada (en metros) asociado a road-quad.",
    )
    parser.add_argument(
        "--line-frac",
        type=float,
        default=0.5,
        help="Posición de la línea de conteo como fracción [0..1] dentro del largo de la vía (0 entrada, 1 salida).",
    )

    parser.add_argument(
        "--ipm-img",
        type=str,
        help=(
            "4 puntos en la imagen (x1,y1,x2,y2,x3,y3,x4,y4) para homografía general: "
            "esquinas del área que queremos 'aplanar'."
        ),
    )
    parser.add_argument(
        "--ipm-world",
        type=str,
        help="4 puntos en el mundo (X1,Y1,...,X4,Y4) correspondientes a ipm-img (en METROS).",
    )

    # Alternativa: calibración 1D metros/píxel
    parser.add_argument(
        "--mpp",
        type=float,
        default=None,
        help="Metros por píxel (si no se usa homografía). No corrige perspectiva, solo da un factor aproximado.",
    )

    # Velocidad / infracción
    parser.add_argument(
        "--speed-window",
        type=float,
        default=1.0,
        help="Ventana (en segundos) para el cálculo de velocidad con historial.",
    )
    parser.add_argument(
        "--speed-limit-kmh",
        type=float,
        required=True,
        help="Límite de velocidad en km/h para marcar infracciones.",
    )

    # Salidas
    parser.add_argument(
        "--save-video",
        type=str,
        default="fusion_out.mp4",
        help="Ruta al video de salida con anotaciones.",
    )
    parser.add_argument(
        "--events-csv",
        type=str,
        default="eventos_aforo.csv",
        help="CSV con eventos de cruce de la línea para aforo.",
    )
    parser.add_argument(
        "--violations-csv",
        type=str,
        default="vel_infracciones.csv",
        help="CSV con infracciones de exceso de velocidad.",
    )
    parser.add_argument(
        "--save-frames-dir",
        type=str,
        default="frames_infracciones",
        help="Directorio para guardar frames de evidencia.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Si se especifica, tambien se guardan recortes (crops) de los vehiculos infractores.",
    )

    return parser.parse_args()


# ---------------- Lógica principal ----------------
def main():
    args = parse_args()

    # Crear carpetas de salida
    frames_dir = Path(args.save_frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = frames_dir / "crops"
    if args.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    # ROI
    roi_polygon = parse_polygon(args.roi) if args.roi else None

    # Puntos del cuadrilátero de la carretera en coordenadas de imagen
    road_quad_pts = None
    if args.road_quad:
        road_quad_pts = np.array(parse_quad(args.road_quad), dtype=np.int32)

    # Modelo
    model = YOLO(args.model)

    # ---------- Captura de video ----------
    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    if not cap.isOpened():
        print("No se pudo abrir la fuente de video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------- Línea de conteo & homografía ----------
    H = None
    ipm_axis_vec = None  # Eje "de avance" en mundo o en IPM

    if args.ipm_img and args.ipm_world:
        img_pts = np.array(parse_quad(args.ipm_img), dtype=np.float32)
        world_pts = np.array(parse_quad(args.ipm_world), dtype=np.float32)
        H, _ = cv2.findHomography(img_pts, world_pts)
        # Eje de avance ~ vector entre el punto cercano medio y el punto lejano medio
        near_center = (world_pts[0] + world_pts[1]) / 2.0
        far_center = (world_pts[2] + world_pts[3]) / 2.0
        axis = far_center - near_center
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            ipm_axis_vec = axis / norm
        else:
            ipm_axis_vec = np.array([0.0, 1.0], dtype=np.float32)

    elif args.road_quad and (args.road_width_m is not None) and (args.road_length_m is not None):
        # Usamos directamente los puntos del cuadrilátero ya parseados
        img_pts = road_quad_pts.astype(np.float32)

        # Construimos un rectángulo en el mundo de tamaño (width x length)
        w = args.road_width_m
        L = args.road_length_m
        world_pts = np.array(
            [
                [0.0, 0.0],
                [w, 0.0],
                [w, L],
                [0.0, L],
            ],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(img_pts, world_pts)

        # Eje de avance ~ eje Y del rectángulo (0,0)->(0,L)
        ipm_axis_vec = np.array([0.0, 1.0], dtype=np.float32)

    # Línea de conteo
    if args.line:
        x1, y1, x2, y2 = args.line
    else:
        # Si tenemos homografía y un rectángulo de carretera, definimos la línea automáticamente
        if H is not None and args.road_quad and (args.road_length_m is not None):
            w = args.road_width_m
            L = args.road_length_m
            frac = np.clip(args.line_frac, 0.0, 1.0)
            y_line = L * frac
            if args.approach == "A->B":
                world_line = np.array([[0.0, y_line], [w, y_line]], dtype=np.float32)
            else:
                world_line = np.array([[w, y_line], [0.0, y_line]], dtype=np.float32)

            H_inv = np.linalg.inv(H)
            homog_world_line = cv2.perspectiveTransform(world_line[None, :, :], H_inv)[0]
            (x1, y1), (x2, y2) = homog_world_line.astype(int).tolist()
        else:
            x1, y1, x2, y2 = 0, height // 2, width, height // 2

    # ---------- Salida de video ----------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))

    # ---------- CSVs ----------
    csv_events_path = Path(args.events_csv)
    f_events = open(csv_events_path, "w", newline="", encoding="utf-8")
    w_events = csv.writer(f_events)
    w_events.writerow(["timestamp", "track_id", "class", "direction", "cx", "cy", "conf", "speed_kmh"])

    csv_viol_path = Path(args.violations_csv)
    f_viol = open(csv_viol_path, "w", newline="", encoding="utf-8")
    w_viol = csv.writer(f_viol)
    w_viol.writerow(["datetime", "track_id", "class", "speed_kmh", "limit_kmh", "cx", "cy"])

    # ---------- Estados para aforo ----------
    last_side_by_id = {}
    side_state_by_id = {}  # tid -> {"first": int, "last": int, "has_pos": bool, "has_neg": bool, "counted": bool}
    counts_total = {"A->B": 0, "B->A": 0}
    counts_by_class = {"A->B": defaultdict(int), "B->A": defaultdict(int)}
    event_count = 0
    first_event_ts = None
    last_event_ts = None

    # ---------- Estados para velocidad ----------
    hist_world = defaultdict(lambda: deque(maxlen=32))  # (t, X, Y) metros
    hist_img = defaultdict(lambda: deque(maxlen=32))  # (t, x, y) píxeles

    # ---------- Bucle principal ----------
    frame_idx = 0
    start_global = time.time()

    try:
        stream = model.track(
            source=args.source,
            conf=args.conf,
            tracker=args.tracker,
            stream=True,
            verbose=False,
        )

        for result in stream:
            frame = result.orig_img
            if frame is None:
                break

            frame_idx += 1
            now = time.time()
            raw = frame.copy()

            # Aplicar ROI si existe
            if roi_polygon is not None:
                roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(roi_mask, [np.array(roi_polygon, dtype=np.int32)], 255)
                frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            else:
                roi_mask = None

            detections = result.boxes
            if detections is None or len(detections) == 0:
                draw_text(frame, f"Límite: {args.speed_limit_kmh:.1f} km/h", (10, 20))
                draw_text(frame, f"Eventos aforo: {event_count}", (10, 40))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                if road_quad_pts is not None:
                    cv2.polylines(frame, [road_quad_pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    for idx, (px, py) in enumerate(road_quad_pts):
                        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
                        draw_text(frame, f"P{idx + 1}", (px + 5, max(10, py - 5)), color=(0, 255, 0), scale=0.5)

                vw.write(frame)
#                if args.show:
 #                   cv2.imshow("Aforo+Velocidad", frame)
  #                  if cv2.waitKey(1) & 0xFF == ord("q"):
   #                     break
                continue

            for box in detections:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                tid = int(box.id[0].item()) if box.id is not None else -1

                name = model.names.get(cls_id, str(cls_id))

                if args.classes_include is not None and name not in args.classes_include:
                    continue

                xA, yA, xB, yB = map(int, box.xyxy[0].tolist())
                cx = (xA + xB) // 2
                cy = yB

                # Filtrado por ROI
                if roi_mask is not None:
                    if roi_mask[cy, cx] == 0:
                        continue
                    inside_roi = True
                else:
                    inside_roi = True

                if not inside_roi:
                    continue

                # Actualizar historial para velocidad
                if H is not None:
                    p_img = np.array([[cx, cy]], dtype=np.float32)[None, :, :]
                    p_world = cv2.perspectiveTransform(p_img, H)[0, 0]
                    Xw, Yw = float(p_world[0]), float(p_world[1])
                    hist_world[tid].append((now, Xw, Yw))
                else:
                    hist_world[tid].clear()

                hist_img[tid].append((now, float(cx), float(cy)))

                v_mps = 0.0
                if H is not None and ipm_axis_vec is not None:
                    v_mps, _, _ = compute_speed_along(hist_world[tid], now, args.speed_window, ipm_axis_vec)
                else:
                    if args.mpp is not None:
                        if args.line:
                            vx = x2 - x1
                            vy = y2 - y1
                            n = math.hypot(vx, vy)
                            if n > 1e-6:
                                axis_unit = np.array([vx / n, vy / n], dtype=np.float32)
                            else:
                                axis_unit = np.array([0.0, 1.0], dtype=np.float32)
                        else:
                            axis_unit = np.array([0.0, -1.0], dtype=np.float32)

                        v_pxps, _, _ = compute_speed_along(hist_img[tid], now, args.speed_window, axis_unit)
                        v_mps = v_pxps * float(args.mpp or 0.0)
                v_kmh = v_mps * 3.6

                # ===== Cruce de línea (aforo) (más robusto) =====
                cur_side = point_line_side(cx, cy, x1, y1, x2, y2)
                prev_side = last_side_by_id.get(tid, cur_side)

                # Estado del track relativo a la línea
                state = side_state_by_id.get(tid)
                if state is None:
                    state = {
                        "first": None,
                        "last": None,
                        "has_pos": False,
                        "has_neg": False,
                        "counted": False,
                    }
                    side_state_by_id[tid] = state

                # Actualizar historial de lados (ignorando exactamente la línea: 0)
                if cur_side != 0:
                    if state["first"] is None:
                        state["first"] = cur_side
                    state["last"] = cur_side
                    if cur_side > 0:
                        state["has_pos"] = True
                    elif cur_side < 0:
                        state["has_neg"] = True

                # Cambio de signo entre frames consecutivos
                sign_change = (
                    prev_side != 0
                    and cur_side != 0
                    and math.copysign(1, prev_side) != math.copysign(1, cur_side)
                )

                crossed = False
                direction = None

                # Caso básico: cambio de signo inmediato
                if (not state["counted"]) and sign_change:
                    crossed = True
                    direction = "A->B" if prev_side < 0 and cur_side > 0 else "B->A"

                # Caso extra: el track ya estuvo a ambos lados de la línea
                elif (not state["counted"]) and state["has_pos"] and state["has_neg"]:
                    crossed = True
                    first = state["first"]
                    last = state["last"] if state["last"] is not None else cur_side
                    if first is not None and last is not None:
                        if first < 0 and last > 0:
                            direction = "A->B"
                        elif first > 0 and last < 0:
                            direction = "B->A"

                if crossed and direction is not None:
                    state["counted"] = True  # nunca volver a contar este track

                    counts_total[direction] += 1
                    counts_by_class[direction][name] += 1
                    w_events.writerow([
                        int(now),
                        tid,
                        name,
                        direction,
                        cx,
                        cy,
                        round(conf, 3),
                        f"{v_kmh:.2f}",
                    ])

                    event_count += 1
                    now_ts = int(now)
                    if first_event_ts is None:
                        first_event_ts = now_ts
                    last_event_ts = now_ts

                last_side_by_id[tid] = cur_side

                # ===== Infracción (exceso) en el sentido de aproximación =====
                crossed_in_approach = crossed and (direction == args.approach)
                color = (255, 128, 0)
                label = f"{name} #{tid} {v_kmh:.1f} km/h"

                if crossed_in_approach and v_kmh > args.speed_limit_kmh and (H is not None or args.mpp is not None):
                    color = (0, 0, 255)
                    label = f"EXCESO {v_kmh:.1f} km/h (> {args.speed_limit_kmh:.0f})"

                    hora_tag = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
                    base = f"{hora_tag}_id{tid}_{name}_{v_kmh:.1f}kmh_gt{int(args.speed_limit_kmh)}"
                    raw_path = frames_dir / f"{base}_raw.jpg"
                    cv2.imwrite(str(raw_path), raw)

                    if args.save_crops:
                        pad_x = int(0.1 * (xB - xA))
                        pad_y = int(0.1 * (yB - yA))
                        xa = max(0, xA - pad_x)
                        ya = max(0, yA - pad_y)
                        xb = min(raw.shape[1] - 1, xB + pad_x)
                        yb = min(raw.shape[0] - 1, yB + pad_y)
                        crop = raw[ya:yb, xa:xb].copy()
                        if crop.size > 0:
                            cv2.imwrite(str(crops_dir / f"{base}_crop.jpg"), crop)

                    w_viol.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                        tid,
                        name,
                        f"{v_kmh:.2f}",
                        f"{args.speed_limit_kmh:.2f}",
                        cx,
                        cy,
                    ])

                # ===== Dibujo bbox =====
                cv2.rectangle(frame, (xA, yA), (xB, yB), color, 2)
                draw_text(frame, label, (xA, max(15, yA - 8)))
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

            # Panel de KPIs (aforo)
            y0 = 28
            draw_text(frame, f"A->B: {counts_total['A->B']}", (10, y0))
            draw_text(frame, f"B->A: {counts_total['B->A']}", (10, y0 + 24))
            y = y0 + 52
            for d in ["A->B", "B->A"]:
                draw_text(frame, f"{d} por clase:", (10, y))
                y += 20
                top = sorted(counts_by_class[d].items(), key=lambda kv: kv[1], reverse=True)[:3]
                for cname, cnt in top:
                    draw_text(frame, f" - {cname}: {cnt}", (10, y))
                    y += 18
                y += 4

            # Dibujo del cuadrilátero de la carretera (road-quad) y sus puntos P1..P4
            if road_quad_pts is not None:
                cv2.polylines(frame, [road_quad_pts], isClosed=True, color=(0, 255, 0), thickness=2)
                for idx, (px, py) in enumerate(road_quad_pts):
                    cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
                    draw_text(
                        frame,
                        f"P{idx + 1}",
                        (px + 5, max(10, py - 5)),
                        color=(0, 255, 0),
                        scale=0.5,
                    )

            # Línea de conteo
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Info de calibración
            calib_text = [f"Limite: {args.speed_limit_kmh:.1f} km/h"]
            if H is not None:
                calib_text.append("Calibracion: HOMOGRAFIA (metros)")
            elif args.mpp is not None:
                calib_text.append(f"Calibracion: {args.mpp:.4f} m/px (1D)")
            else:
                calib_text.append("Calibracion: SIN metros (solo conteo)")

            for i, txt in enumerate(calib_text):
                draw_text(frame, txt, (10, 10 + 18 * i))

            vw.write(frame)

#            if args.show:
 #               cv2.imshow("Aforo+Velocidad", frame)
  #              if cv2.waitKey(1) & 0xFF == ord("q"):
   #                 break

    finally:
        end_global = time.time()
        total_time = max(1e-6, end_global - start_global)

        if event_count > 0 and first_event_ts is not None and last_event_ts is not None and last_event_ts > first_event_ts:
            dur_used_s = last_event_ts - first_event_ts
        else:
            dur_used_s = total_time
        dur_used_s = max(dur_used_s, 1e-6)

        veh_ab = counts_total["A->B"]
        veh_ba = counts_total["B->A"]

        veh_per_sec_ab = veh_ab / dur_used_s
        veh_per_sec_ba = veh_ba / dur_used_s

        veh_per_min_ab = veh_per_sec_ab * 60.0
        veh_per_min_ba = veh_per_sec_ba * 60.0

        veh_per_h_ab = veh_per_min_ab * 60.0
        veh_per_h_ba = veh_per_min_ba * 60.0

        print("\n===== RESUMEN AFORO =====")
        print(f"Duración observada (s): {dur_used_s:.1f}")
        print(f"A->B: {veh_ab} veh  ({veh_per_min_ab:.2f} veh/min, {veh_per_h_ab:.2f} veh/h)")
        print(f"B->A: {veh_ba} veh  ({veh_per_min_ba:.2f} veh/min, {veh_per_h_ba:.2f} veh/h)")

        # CSV resumen global
        resumen_path = Path("resumen_aforo.csv")
        with open(resumen_path, "w", newline="", encoding="utf-8") as fsum:
            wsum = csv.writer(fsum)
            wsum.writerow(["direction", "total", "veh_per_min", "veh_per_hour", "duration_s"])
            wsum.writerow(["A->B", veh_ab, f"{veh_per_min_ab:.4f}", f"{veh_per_h_ab:.4f}", f"{dur_used_s:.2f}"])
            wsum.writerow(["B->A", veh_ba, f"{veh_per_min_ba:.4f}", f"{veh_per_h_ba:.4f}", f"{dur_used_s:.2f}"])

        # CSV resumen por clase
        resumen_class_path = Path("resumen_aforo_por_clase.csv")
        with open(resumen_class_path, "w", newline="", encoding="utf-8") as fsc:
            wsc = csv.writer(fsc)
            wsc.writerow(["direction", "class", "total"])
            for direction in ["A->B", "B->A"]:
                for cname, cnt in counts_by_class[direction].items():
                    wsc.writerow([direction, cname, cnt])

        # Cierre
        f_events.close()
        f_viol.close()
        vw.release()
        cap.release()
        cv2.destroyAllWindows()

        print(f"\nVideo anotado:              {Path(args.save_video).resolve()}")
        print(f"Eventos aforo (CSV):        {csv_events_path.resolve()}")
        print(f"Infracciones (CSV):         {csv_viol_path.resolve()}")
        print(f"Resumen aforo (CSV):        {resumen_path.resolve()}")
        print(f"Resumen por clase (CSV):    {resumen_class_path.resolve()}")
        print(f"Frames evidencia:           {frames_dir.resolve()}")


if __name__ == "__main__":
    main()
