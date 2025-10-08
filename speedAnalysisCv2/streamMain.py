import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# --------------------------
# Configuración inicial
# --------------------------
st.set_page_config(page_title="YOLO + Supervision Demo", layout="wide")
st.title("🎥 Detección con YOLOv8 + Supervision + Streamlit")

# Cargar el modelo YOLO
model = YOLO("traffic_analysis.pt")

# Inicializar el tracker y los anotadores
tracker = sv.ByteTrack()
box_annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# --------------------------
# Función de procesamiento
# --------------------------
def process_frame(frame: np.ndarray) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    annotated_frame = trace_annotator.annotate(annotated_frame, detections=detections)

    return annotated_frame

# --------------------------
# Subir o elegir video
# --------------------------
video_file = st.file_uploader("Sube un video o usa el predeterminado", type=["mp4", "mov", "avi"])

if video_file is not None:
    tfile = "temp_video.mp4"
    with open(tfile, "wb") as f:
        f.write(video_file.read())
    video_path = tfile
else:
    video_path = "../sampleVideos/fuentesBethoven.mp4"

# --------------------------
# Mostrar video procesado
# --------------------------
cap = cv2.VideoCapture(video_path)
stframe = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated_frame = process_frame(frame)

    stframe.image(annotated_frame, channels="RGB")

cap.release()
st.success("✅ Video finalizado")
