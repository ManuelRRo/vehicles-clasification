import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt

# --------------------------
# Configuración inicial
# --------------------------
st.set_page_config(page_title="YOLO + Supervision Demo", layout="wide")
st.title("🎥 Detección con YOLOv8 + Supervision + Streamlit")

# Cargar el modelo YOLO
model = YOLO("traffic_analysis.pt")

# Inicializar el tracker y los anotadores
tracker = sv.ByteTrack()

polygon =  np.array([[832, 919], [822, 635], [916, 635], [932, 929]])
#polygon2 = np.array([[619, 307], [625, 434], [952, 437], [937, 309]])
#polygon2 = np.array([[1219, 557], [1225, 934], [1852, 937], [1837, 559]])
polygon2 = np.array([[1212, 502], [1206, 700], [1794, 748], [1791, 555]])
#polygon3 = np.array([[908, 319], [977, 319],[975, 11], [889, 9]])
polygon3 = np.array([[845, 712], [1145, 745], [1166, 475], [866, 450]])
zone = sv.PolygonZone(polygon=polygon)
zone2 = sv.PolygonZone(polygon=polygon2)
zone3 = sv.PolygonZone(polygon=polygon3)

box_annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1)
zone_annotator2 = sv.PolygonZoneAnnotator(zone=zone2, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1) 
zone_annotator3 = sv.PolygonZoneAnnotator(zone=zone3, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1)


# --------------------------
# Función de procesamiento
# --------------------------
def process_frame(frame: np.ndarray) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    detections_in_polygon_1 = zone.trigger(detections=detections)
    detections_in_polygon_2 = zone2.trigger(detections=detections)
    detections_in_polygon_3 = zone3.trigger(detections=detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    annotated_frame = trace_annotator.annotate(annotated_frame, detections=detections)
    annotated_frame = zone_annotator.annotate(scene=annotated_frame)
    annotated_frame = zone_annotator2.annotate(scene=annotated_frame)
    annotated_frame = zone_annotator3.annotate(scene=annotated_frame)

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
frame_generator = sv.get_video_frames_generator(source_path=video_path)
for frame in frame_generator:
    annotated_frame = process_frame(frame)
    stframe.image(annotated_frame, channels="RGB")

cap.release()
st.success("✅ Video finalizado")

