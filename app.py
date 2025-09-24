# app.py
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import supervision as sv

# ---------- Parámetros del ejemplo ----------
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [0, TARGET_HEIGHT - 1],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# ---------------------- UI Streamlit ---------------------------
st.set_page_config(page_title="YOLO + Supervision en Streamlit", layout="wide")
st.title("Streaming de video con YOLO + Supervision (ByteTrack)")

with st.sidebar:
    st.header("Parámetros")
    uploaded = st.file_uploader("Sube un video (MP4/MOV/AVI)", type=["mp4", "mov", "avi"])
    default_path = "ranchoNavarra.mp4"
    source_path_text = st.text_input("…o ingresa ruta local", value=default_path)

    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.05)
    iou_threshold = st.slider("IoU threshold (NMS)", 0.0, 1.0, 0.7, 0.05)
    model_name = st.selectbox("Modelo YOLO", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)

    save_output = st.checkbox("Guardar video de salida (result.mp4)")
    max_fps_display = st.slider("FPS máximos al mostrar", 5, 60, 30, 1)

    start = st.button("Iniciar procesamiento")

if uploaded:
    tmp_path = Path("uploaded_video.mp4")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())
    source_video_path = str(tmp_path)
else:
    source_video_path = source_path_text

frame_placeholder = st.empty()
status_placeholder = st.empty()
progress_bar = st.progress(0, text="Preparando…")

if start:
    try:
        video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    except Exception as e:
        st.error(f"No se pudo abrir el video: {e}")
        st.stop()

    model = YOLO(model_name)

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps,
        track_activation_threshold=confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    sink = sv.VideoSink("result.mp4", video_info) if save_output else None
    total_frames = max(1, video_info.total_frames)
    frame_idx = 0

    try:
        if sink:
            sink.__enter__()

        for frame in frame_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    y_end = coordinates[tracker_id][0]
                    y_start = coordinates[tracker_id][-1]
                    distance = abs(y_start - y_end)
                    t = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / t * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            annotated = frame.copy()
            annotated = trace_annotator.annotate(scene=annotated, detections=detections)
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

            if sink:
                sink.write_frame(annotated)

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0),
                                  text=f"Procesando frame {frame_idx}/{total_frames}")

            time.sleep(max(0, 1.0 / max_fps_display))

        status_placeholder.success("✅ Procesamiento terminado.")
        if sink:
            st.video("result.mp4")

    finally:
        if sink:
            sink.__exit__(None, None, None)
        progress_bar.empty()
