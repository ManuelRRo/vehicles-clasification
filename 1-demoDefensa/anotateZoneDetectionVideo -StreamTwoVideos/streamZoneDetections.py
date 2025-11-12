# app.py
import time
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO

# ------------------ Configuración base ------------------
SOURCE_VIDEO_PATH = "fuentesBethoven17seg.mp4"   # video izquierdo
MODEL_PATH = "traffic_analysis.pt"               # tu modelo YOLO

# Polígonos (ajústalos a tu escena)
POLYGON_ZONE1 = np.array([[954, 789], [960, 1074], [1085, 1071], [1070, 786]])
POLYGON_ZONE2 = np.array([[1132, 496], [1147, 674], [1770, 656], [1749, 463]])
POLYGON_ZONE3 = np.array([[933, 457], [829, 454], [832, 6], [927, 9]])

st.set_page_config(page_title="Traffic Stream", layout="wide")

# ------------------ Carga de herramientas ------------------
@st.cache_resource
def load_tools():
    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack()
    box_annotator = sv.RoundBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    z1 = sv.PolygonZone(polygon=POLYGON_ZONE1)
    z2 = sv.PolygonZone(polygon=POLYGON_ZONE2)
    z3 = sv.PolygonZone(polygon=POLYGON_ZONE3)

    za1 = sv.PolygonZoneAnnotator(zone=z1, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1)
    za2 = sv.PolygonZoneAnnotator(zone=z2, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1)
    za3 = sv.PolygonZoneAnnotator(zone=z3, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1)

    return model, tracker, box_annotator, label_annotator, trace_annotator, (z1, z2, z3), (za1, za2, za3)

model, tracker, box_annotator, label_annotator, trace_annotator, (zone1, zone2, zone3), (za1, za2, za3) = load_tools()

# ------------------ UI: Barra lateral y columnas ------------------
st.title("🚦 Análisis de tráfico (YOLO + Supervision)")

st.sidebar.markdown("**Fuente de video 1 (izquierda):**")
st.sidebar.text(SOURCE_VIDEO_PATH)

SOURCE_VIDEO_PATH2 = st.sidebar.text_input("Fuente de video 2 (derecha)", "video_2.mp4")

start_btn = st.sidebar.button("▶️ Iniciar / Reiniciar")
stop_btn  = st.sidebar.button("⏹️ Detener")

progress_bar_left  = st.sidebar.progress(0, text="Preparando video 1…")
progress_bar_right = st.sidebar.progress(0, text="Preparando video 2…")

if "zone_counts" not in st.session_state:
    st.session_state.zone_counts = {"Zona 1": 0, "Zona 2": 0, "Zona 3": 0}

st.sidebar.markdown("### 🚗 Conteo por zona (en tiempo real)")
counts_ph = st.sidebar.empty()

c_left, c_right = st.columns(2)
frame_area_left  = c_left.empty()
frame_area_right = c_right.empty()

# ------------------ Estado de ejecución ------------------
if "running" not in st.session_state:
    st.session_state.running = False
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# ------------------ Utilidad: dibujar cajas de info ------------------
def draw_info_boxes(img, zone1_count, zone2_count, zone3_count):
    # Caja 1 (Calle 3)
    info_title = "Calle 3"
    info_vehicles = f"Vehiculos: {zone3_count}"
    x, y = 500, 30
    width, height = 280, 100
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), -1)
    cv2.rectangle(img, (x + 2, y + 2), (x + width - 2, y + height - 2), (255, 255, 255), -1)
    cv2.putText(img, info_title, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, info_title, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, info_vehicles, (x + 10, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Caja 2 (Calle 2)
    info_title = "Calle 2"
    info_vehicles = f"Vehiculos: {zone2_count}"
    x, y = 1500, 320
    width, height = 280, 100
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), -1)
    cv2.rectangle(img, (x + 2, y + 2), (x + width - 2, y + height - 2), (255, 255, 255), -1)
    cv2.putText(img, info_title, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, info_title, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, info_vehicles, (x + 10, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Caja 3 (Calle 1)
    info_title = "Calle 1"
    info_vehicles = f"Vehiculos: {zone1_count}"
    x, y = 500, 700
    width, height = 280, 100
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), -1)
    cv2.rectangle(img, (x + 2, y + 2), (x + width - 2, y + height - 2), (255, 255, 255), -1)
    cv2.putText(img, info_title, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, info_title, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, info_vehicles, (x + 10, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return img

# ------------------ Bucle principal ------------------
if st.session_state.running:

    cap1 = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    cap2 = cv2.VideoCapture(SOURCE_VIDEO_PATH2)

    if not cap1.isOpened():
        st.error(f"No se pudo abrir el video 1: {SOURCE_VIDEO_PATH}")
        st.stop()

    if not cap2.isOpened():
        c_right.warning(f"No se pudo abrir el video 2: {SOURCE_VIDEO_PATH2}. Mostrando placeholder…")

    # FPS del video 1
    try:
        fps = max(1, int(sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH).fps))
    except Exception:
        fps = 25

    # Total frames del video 1 (para la barra de progreso izquierda)
    try:
        vi1 = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
        total_frames_left = vi1.total_frames or 1
    except Exception:
        total_frames_left = 1

    # Si el video 2 es archivo, estimar frames para barra
    try:
        vi2 = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH2)
        total_frames_right = vi2.total_frames or 1
        right_is_file = True
    except Exception:
        total_frames_right = 1
        right_is_file = False

    frame_idx_left = 0
    frame_idx_right = 0

    while st.session_state.running:
        ok1, frame1 = cap1.read()
        ok2, frame2 = cap2.read() if cap2.isOpened() else (False, None)

        # Finaliza si el video 1 ya no entrega frames
        if not ok1:
            st.session_state.running = False
            break

        # ======== Pipeline Video 1 (IZQUIERDA, con anotaciones) ========
        results = model(frame1, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # IDs de tracker seguros
        n = len(detections)
        tracker_ids = detections.tracker_id
        tracker_ids = [None] * n if tracker_ids is None else list(tracker_ids)

        # Disparar zonas
        zone1.trigger(detections=detections)
        zone2.trigger(detections=detections)
        zone3.trigger(detections=detections)

        # Anotaciones
        annotated = box_annotator.annotate(frame1.copy(), detections=detections)
        annotated = za1.annotate(scene=annotated)
        annotated = za2.annotate(scene=annotated)
        annotated = za3.annotate(scene=annotated)
        annotated = trace_annotator.annotate(annotated, detections=detections)

        # Cajas de información
        annotated = draw_info_boxes(
            annotated,
            zone1_count=zone1.current_count,
            zone2_count=zone2.current_count,
            zone3_count=zone3.current_count
        )

        rgb_left = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_area_left.image(rgb_left, use_container_width=True)

        # ======== Video 2 (DERECHA): mostrar “raw” ========
        if ok2 and frame2 is not None:
            rgb_right = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame_area_right.image(rgb_right, use_container_width=True)
            frame_idx_right += 1
        else:
            c_right.info("Sin señal en el Video 2.")

        # Actualización de conteos y UI
        frame_idx_left += 1
        st.session_state.zone_counts["Zona 1"] = zone1.current_count
        st.session_state.zone_counts["Zona 2"] = zone2.current_count
        st.session_state.zone_counts["Zona 3"] = zone3.current_count

        counts_ph.table([
            {"Zona": "Zona 1", "Conteo": st.session_state.zone_counts["Zona 1"]},
            {"Zona": "Zona 2", "Conteo": st.session_state.zone_counts["Zona 2"]},
            {"Zona": "Zona 3", "Conteo": st.session_state.zone_counts["Zona 3"]},
        ])

        progress_bar_left.progress(
            min(frame_idx_left / total_frames_left, 1.0),
            text=f"Video 1: frame {frame_idx_left}/{total_frames_left}"
        )

        if right_is_file and total_frames_right > 1:
            progress_bar_right.progress(
                min(frame_idx_right / total_frames_right, 1.0),
                text=f"Video 2: frame {frame_idx_right}/{total_frames_right}"
            )
        else:
            progress_bar_right.progress(0.0, text="Video 2 reproduciéndose…")

        time.sleep(1.0 / fps)

    cap1.release()
    if cap2.isOpened():
        cap2.release()

    st.success("Reproducción finalizada.")
else:
    c_left.info("Haz clic en **Iniciar / Reiniciar** para comenzar.")
    c_right.info("Configura la **Fuente de video 2** en la barra lateral.")
