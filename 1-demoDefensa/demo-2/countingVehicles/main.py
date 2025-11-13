import time
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import json

SOURCE_VIDEO_PATH = "../../resources/videos/gabrielaMistralNoAerea.mp4"
MODEL_PATH = "../../resources/models/traffic_analysis.pt"
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
    
    return tracker, box_annotator, label_annotator, trace_annotator

tracker, box_annotator, label_annotator, trace_annotator = load_tools()

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
    
#    st.markdown("### 🚗 Conteo por zona (en tiempo real)")
    counts_ph = st.empty()

cross_counts_one = defaultdict(int)
cross_counts_two = defaultdict(int)
cross_counts_three = defaultdict(int)
cross_counts_four = defaultdict(int)
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

        annotated = box_annotator.annotate(frame.copy(), detections=detections)
        #annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)
        annotated = trace_annotator.annotate(annotated, detections=detections)

        # Diccionario para acumular conteos por clase

        
        ## counting lines
        crossed_in, crossed_out = line_zone.trigger(detections)
        crossed_in2, crossed_out2 = line_zone2.trigger(detections)
        crossed_in3, crossed_out3 = line_zone3.trigger(detections)

        detections_in_line_one = detections[crossed_in]
        detections_in_line_two = detections[crossed_in2]
        detections_out_line_two = detections[crossed_out2]
        detections_in_line_four = detections[crossed_in3]
        
        class_id_test_one = detections_in_line_one.class_id
        class_id_test_two = detections_in_line_two.class_id
        class_id_test_three = detections_out_line_two.class_id
        class_id_test_four = detections_in_line_four.class_id

        if len(class_id_test_one) > 0:
            for cls_id in class_id_test_one:
                cls_id = int(cls_id)
                # nombre de la clase según el modelo
                class_name = model.model.names[cls_id]

                # acumular conteo
                cross_counts_one[class_name] += 1

                # imprimir detección en tiempo real
                print(f"Detección out: class {cls_id} ({class_name})")
        
        if len(class_id_test_two) > 0:
            for cls_id in class_id_test_two:
                cls_id = int(cls_id)
                # nombre de la clase según el modelo
                class_name = model.model.names[cls_id]

                # acumular conteo
                cross_counts_two[class_name] += 1

                # imprimir detección en tiempo real
                print(f"Detección out: class {cls_id} ({class_name})")        

        if len(class_id_test_three) > 0:
            for cls_id in class_id_test_three:
                cls_id = int(cls_id)
                # nombre de la clase según el modelo
                class_name = model.model.names[cls_id]

                # acumular conteo
                cross_counts_three[class_name] += 1

                # imprimir detección en tiempo real
                print(f"Detección out: class {cls_id} ({class_name})")

        if len(class_id_test_four) > 0:
            for cls_id in class_id_test_four:
                cls_id = int(cls_id)
                # nombre de la clase según el modelo
                class_name = model.model.names[cls_id]

                # acumular conteo
                cross_counts_four[class_name] += 1

                # imprimir detección en tiempo real
                print(f"Detección out: class {cls_id} ({class_name})")

        print("Resumen acumulado de cruces:", dict(cross_counts_one))
        print("Resumen acumulado de cruces:", dict(cross_counts_four))
        print("Resumen acumulado de cruces:", dict(cross_counts_three))
        print("Resumen acumulado de cruces:", dict(cross_counts_two))
        # Line zone annotators
        line_zone_annotator.annotate(annotated, line_counter=line_zone)
        line_zone_annotator.annotate(annotated, line_counter=line_zone2)
        line_zone_annotator.annotate(annotated, line_counter=line_zone3)
        
        info_title = f"Resumen de vehiculos"

        info_vehicles1 = f"Norte -> Sur: {line_zone.in_count}" 
        info_vehicles2  = f"Norte <- Sur: {line_zone3.in_count}" 
        info_vehicles3 = f"Este -> Oeste: {line_zone2.out_count}" 
        info_vehicles4 = f"Este <- Oeste: {line_zone2.in_count}" 

       
        x, y = 30, 30
        width, height = 390, 150  # tamaño del rectángulo

        # Fondo blanco con borde verde
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), -1)  # relleno verde

        cv2.rectangle(annotated, (x + 2, y + 2), (x + width - 2, y + height - 2), (255, 255, 255), -1)  # interior blanco

        # Escribir texto encima (BGR)
        cv2.putText(annotated, info_title, (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)  # título
        
        cv2.putText(annotated, info_title, (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(annotated, info_vehicles1, (x + 10, y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        cv2.putText(annotated, info_vehicles2, (x + 10, y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        cv2.putText(annotated, info_vehicles3, (x + 10, y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        cv2.putText(annotated, info_vehicles4, (x + 10, y + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        info_text1 = f"Norte -> Sur: " 
        info_text2  = f"Norte <- Sur:" 
        info_text3 = f"Este -> Oeste:" 
        info_text4 = f"Este <- Oeste:" 
        
        x, y = 50,619
        
        cv2.putText(annotated, info_text1, (x , y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # título

        x, y = 29, 368
        
        cv2.putText(annotated, info_text2, (x , y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # título
        
        x, y = 622, 653
        
        cv2.putText(annotated, info_text3, (x , y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # título

        x, y = 876, 562
        
        cv2.putText(annotated, info_text4, (x , y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # título

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_area.image(rgb, use_container_width=True)

        frame_idx += 1
   
        progress_bar.progress(min(frame_idx / total_frames, 1.0),
                                  text=f"Procesando frame {frame_idx}/{total_frames}")
        
        time.sleep(1.0 / fps)

    merged = {
    "zone1": dict(cross_counts_one),
    "zone2": dict(cross_counts_two),
    "zone3": dict(cross_counts_three),
    "zone4": dict(cross_counts_four),
    }

    txt = st.text_area(
    "Prompt Para analisis",
    f"""
    En base a la información presentada, puedes recomendar cambios en los tiempos de fase de los semáforos.

    Conteo por fases:
    {json.dumps({
        "Norte a Sur": cross_counts_one,
        "Sur a Norte": cross_counts_two,
        "Este a Oeste": cross_counts_three,
        "Oeste a Este": cross_counts_four
    }, indent=4)}
    """,
    )

#    df = pd.DataFrame(resumen_final).fillna(0).astype(int).T
#    st.table(df)
    cap.release()
    st.success("Video finalizado.")
    
else:
    c1.info("Haz clic en **Iniciar / Reiniciar** para comenzar.")
