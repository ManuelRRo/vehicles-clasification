# app.py
import time
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import supervision as sv
from trafficCalculationTime.algorythm import detect_and_count_vehicles
from ultralytics import YOLO

SOURCE_VIDEO_PATH = "../../../resources/videos/gabrielaMistralNoAerea.mp4"
MODEL_PATH = "../../../resources/models/traffic_analysis.pt"
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(generator)
model = YOLO(MODEL_PATH)
POLYGON_ZONE1 = np.array([[954, 789], [960, 1074], [1085, 1071], [1070, 786]])
POLYGON_ZONE2 = np.array([[1132, 496], [1147, 674], [1770, 656], [1749, 463]])
POLYGON_ZONE3 = np.array([[933, 457], [829, 454], [832, 6], [927, 9]])

# LINES
START_LINE1 = sv.Point(308, 410)
END_LINE1 = sv.Point(367, 562)

START_LINE2 = sv.Point(697, 562)
END_LINE2 = sv.Point(856, 505)

START_LINE3 = sv.Point(725, 328)
END_LINE3 = sv.Point(614, 226)


# LINES ZONES
line_zone = sv.LineZone(start=START_LINE1, end=END_LINE1)
line_zone2 = sv.LineZone(start=START_LINE2, end=END_LINE2)
line_zone3 = sv.LineZone(start=START_LINE3, end=END_LINE3)

# Line zone annotator
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_thickness=4,
    text_scale=0.8)

# Count Acumulator



st.set_page_config(page_title="Traffic Stream", layout="wide")
frame_idx = 0
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
total_frames = video_info.total_frames if video_info.total_frames is not None else 1

@st.cache_resource
def load_tools():

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

st.title("🚦 Anlisis de tráfico (YOLO + Supervision)")
c1, c2 = st.columns([3, 1])
with c2:
    st.markdown("**Fuente de video:**")
    st.text(SOURCE_VIDEO_PATH)
    start_btn = st.button("▶️ Iniciar / Reiniciar")
    stop_btn = st.button("⏹️ Detener")
    progress_bar = st.progress(0, text="Preparando…")
    if "zone_counts" not in st.session_state:
        st.session_state.zone_counts = {"Zona 1": 0, "Zona 2": 0, "Zona 3": 0}
    
    st.markdown("### 🚗 Conteo por zona (en tiempo real)")
    counts_ph = st.empty()


if "running" not in st.session_state:
    st.session_state.running = False
if start_btn: st.session_state.running = True
if stop_btn:  st.session_state.running = False

frame_area = c1.empty()

if st.session_state.running:
    
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    if not cap.isOpened():
        st.error(f"No se pudo abrir el video: {SOURCE_VIDEO_PATH}")
        st.stop()

    try:
        fps = max(1, int(sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH).fps))
    except Exception:
        fps = 25

    while st.session_state.running:
        ok, frame = cap.read()
        if not ok:
            st.session_state.running = False
            break

        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # ---- FIX: construir tracker_ids seguro (sin usar "or []") ----
        n = len(detections)
        tracker_ids = detections.tracker_id
        if tracker_ids is None:
            tracker_ids = [None] * n
        else:
            # Asegurar que tenga el mismo largo y sea indexable
            tracker_ids = list(tracker_ids)

        labels = []
        for cid, tid in zip(detections.class_id.astype(int), tracker_ids):
            cls_name = results.names[int(cid)] if hasattr(results, "names") else str(int(cid))
            prefix = f"#{int(tid)} " if tid is not None else ""
            labels.append(f"{prefix}{cls_name}")

        zone1.trigger(detections=detections)
        zone2.trigger(detections=detections)
        zone3.trigger(detections=detections)

        annotated = box_annotator.annotate(frame.copy(), detections=detections)
        #annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)
        annotated = za1.annotate(scene=annotated)
        annotated = za2.annotate(scene=annotated)
        annotated = za3.annotate(scene=annotated)
        annotated = trace_annotator.annotate(annotated, detections=detections)
        
        ## counting lines
        crossed_in, crossed_out = line_zone.trigger(detections)
        crossed_in2, crossed_out2 = line_zone2.trigger(detections)
        crossed_in3, crossed_out3 = line_zone3.trigger(detections)
        

        # Line zone annotators
        line_zone_annotator.annotate(annotated, line_counter=line_zone)
        line_zone_annotator.annotate(annotated, line_counter=line_zone2)
        line_zone_annotator.annotate(annotated, line_counter=line_zone3)
        # # ======= AQUI ESCRIBES EL TEXTO EN EL FRAME (BGR) =======
        # texto1 = f"Zona 1: {zone1.current_count}"
        # texto2 = f"Zona 2: {zone2.current_count}"
        # texto3 = f"Zona 3: {zone3.current_count}"

        # cv2.putText(
        #     annotated, texto1,
        #     (20, 40),                      # posición (x, y)
        #     cv2.FONT_HERSHEY_SIMPLEX,      # fuente
        #     1.0,                           # tamaño
        #     (0, 255, 0),                   # color BGR (verde)
        #     2,                             # grosor
        #     cv2.LINE_AA                    # tipo de línea
        # )

        # cv2.putText(annotated, texto2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(annotated, texto3, (600, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        info_title = "Calle 3"
        info_vehicles = f"Vehiculos: {zone3.current_count}"
        
        x, y = 500, 30
        width, height = 280, 100  # tamaño del rectángulo

        # Fondo blanco con borde verde
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), -1)  # relleno verde
        cv2.rectangle(annotated, (x + 2, y + 2), (x + width - 2, y + height - 2), (255, 255, 255), -1)  # interior blanco

        # Escribir texto encima (BGR)
        cv2.putText(annotated, info_title, (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)  # título
        cv2.putText(annotated, info_title, (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(annotated, info_vehicles, (x + 10, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        info_title = "Calle 2"
        info_vehicles = f"Vehiculos: {zone2.current_count}"
        
        x, y = 1500, 320
        width, height = 280, 100  # tamaño del rectángulo

        # Fondo blanco con borde verde
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), -1)  # relleno verde
        cv2.rectangle(annotated, (x + 2, y + 2), (x + width - 2, y + height - 2), (255, 255, 255), -1)  # interior blanco

        # Escribir texto encima (BGR)
        cv2.putText(annotated, info_title, (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)  # título
        cv2.putText(annotated, info_title, (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(annotated, info_vehicles, (x + 10, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        info_title = "Calle 1"
        info_vehicles = f"Vehiculos: {zone1.current_count}"
        
        x, y = 500, 700
        width, height = 280, 100  # tamaño del rectángulo

        # Fondo blanco con borde verde
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), -1)  # relleno verde
        cv2.rectangle(annotated, (x + 2, y + 2), (x + width - 2, y + height - 2), (255, 255, 255), -1)  # interior blanco

        # Escribir texto encima (BGR)
        cv2.putText(annotated, info_title, (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)  # título
        cv2.putText(annotated, info_title, (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(annotated, info_vehicles, (x + 10, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        
        

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_area.image(rgb, use_container_width=True)

        frame_idx += 1

        st.session_state.zone_counts["Zona 1"] = zone1.current_count
        st.session_state.zone_counts["Zona 2"] = zone2.current_count
        st.session_state.zone_counts["Zona 3"] = zone3.current_count

        counts_ph.table([
        {"Zona": "Zona 1", "Conteo": st.session_state.zone_counts["Zona 1"]},
        {"Zona": "Zona 2", "Conteo": st.session_state.zone_counts["Zona 2"]},
        {"Zona": "Zona 3", "Conteo": st.session_state.zone_counts["Zona 3"]},
        ])
        
        progress_bar.progress(min(frame_idx / total_frames, 1.0),
                                  text=f"Procesando frame {frame_idx}/{total_frames}")
        
        time.sleep(1.0 / fps)

    cap.release()
    st.success("Video finalizado.")
else:
