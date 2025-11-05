from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import os
import cv2
from collections import Counter
HOME = os.getcwd()
SOURCE_VIDEO_PATH = f"{HOME}/proceres.mp4"
SOURCE_VIDEO_PATH2 = f"{HOME}/frames/frameProceres.png"
SOURCE_VIDEO_PATH3 = f"{HOME}/frames/bethoven2.png"

SOURCE_IMG_PATH_1 = f"{HOME}/frames/bethoven1.png"
SOURCE_IMG_PATH_2 = f"{HOME}/frames/bethoven2.png"
SOURCE_IMG_PATH_3 = f"{HOME}/frames/bethoven3.png"

#model = YOLO("yolov8n.pt")
model = YOLO("traffic_analysis.pt")
print(f"model.names {model.names}")

tracker = sv.ByteTrack()
# initiate polygon zone
polygon = np.array([[699, 549], [768, 549], [778, 760], [696, 760]])
polygon2 = np.array([[830, 465], [1276, 456], [1269, 334], [821, 351]])
polygon3 = np.array([[608, 319], [677, 319], [675, 11], [589, 9]])

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
zone = sv.PolygonZone(polygon=polygon)
zone2 = sv.PolygonZone(polygon=polygon2)
zone3 = sv.PolygonZone(polygon=polygon3)
# initiate annotators
box_annotator = sv.RoundBoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=1)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1)
zone_annotator2 = sv.PolygonZoneAnnotator(zone=zone2, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1) 
zone_annotator3 = sv.PolygonZoneAnnotator(zone=zone3, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1)
# extract video frame
# generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# iterator = iter(generator)
# frame = next(iterator)

# frame2 = cv2.imread(SOURCE_VIDEO_PATH3)

# # detect
# results = model(frame2)[0]
# detections = sv.Detections.from_ultralytics(results)
# detections = tracker.update_with_detections(detections)
# detections_in_polygon_1 = zone.trigger(detections=detections)
# detections_in_polygon_2 = zone2.trigger(detections=detections)
# detections_in_polygon_3 = zone3.trigger(detections=detections)


# # print(f"detecciones en results {detections} \n")
# # print(f"detecciones en zona \n {detections_in_polygon_1} ")
# # print(f"detecciones en zona \n {detections_in_polygon_2} ")
# # print(f"detecciones en zona \n {detections_in_polygon_3} ")

# # annotate
# labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]
# frame2 = box_annotator.annotate(scene=frame2, detections=detections)
# #frame2 = label_annotator.annotate(scene=frame2, detections=detections, labels=labels)
# frame2 = zone_annotator.annotate(scene=frame2)
# frame2 = zone_annotator2.annotate(scene=frame2)
# frame2 = zone_annotator3.annotate(scene=frame2)

# print(f"Current count in zone 1: {zone.current_count}")
# print(f"Current count in zone 2: {zone2.current_count}")
# print(f"Current count in zone 3: {zone3.current_count}")
# sv.plot_image(frame2)
# plt.savefig("resultado.png")
# ##############################################################################################
# det_z1 = detections[detections_in_polygon_1]
# det_z2 = detections[detections_in_polygon_2]
# det_z3 = detections[detections_in_polygon_3]

# # === NUEVO: nombres de clase por zona ===
# classes_z1 = [model.names[int(c)] for c in det_z1.class_id]
# classes_z2 = [model.names[int(c)] for c in det_z2.class_id]
# classes_z3 = [model.names[int(c)] for c in det_z3.class_id]

# print("Clases en zona 1:", classes_z1)
# print("Clases en zona 2:", classes_z2)
# print("Clases en zona 3:", classes_z3)

# print("Conteo por clase (zona 2):", Counter(classes_z2))

def count_classes_in_zone(
    image: np.ndarray,
    zone_or_polygon,
    model,
    tracker: sv.ByteTrack,
    *,
    return_annotated: bool = False,
    reset_tracker: bool = True
):
    """
    image: np.ndarray (BGR de OpenCV)
    zone_or_polygon: sv.PolygonZone  O  iterable de puntos [[x1,y1], [x2,y2], ...]
    model: modelo YOLO de ultralytics ya cargado
    tracker: instancia de sv.ByteTrack
    return_annotated: si True, devuelve imagen con cajas y polígono
    reset_tracker: resetea el tracker (útil para imágenes sueltas)

    return: (counts_dict, annotated_image_or_None)
    """
    # Asegurar zona
    if isinstance(zone_or_polygon, sv.PolygonZone):
        zone = zone_or_polygon
    else:
        polygon = np.array(zone_or_polygon, dtype=int)
        zone = sv.PolygonZone(polygon=polygon)

    # (opcional) reset si procesas imágenes sueltas y no un flujo de video
    if reset_tracker and hasattr(tracker, "reset"):
        tracker.reset()

    # Inferencia -> Detections -> Track
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # Máscara de detecciones dentro de la zona
    in_zone_mask = zone.trigger(detections=detections)

    # Filtrar detecciones que caen dentro de la zona
    det_in_zone = detections[in_zone_mask]

    # Nombres de clase
    classes_in_zone = [model.names[int(c)] for c in det_in_zone.class_id]
    counts = dict(Counter(classes_in_zone))

    annotated = None
    if return_annotated:
        # Anotar solo lo que está en la zona (puedes cambiar a 'detections' si quieres todo)
        frame = image.copy()
        box_annot = sv.RoundBoxAnnotator(thickness=1)
        zone_annot = sv.PolygonZoneAnnotator(
            zone=zone, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1
        )
        frame = box_annot.annotate(scene=frame, detections=det_in_zone)
        frame = zone_annot.annotate(scene=frame)
        annotated = frame

    return counts, annotated

frame2 = cv2.imread(SOURCE_VIDEO_PATH3)
# Usa un polígono existente (p. ej., polygon2) o pasa la sv.PolygonZone ya creada (zone2)
counts_z2, annotated_z2 = count_classes_in_zone(
    frame2,
    polygon,          # o zone2
    model,
    tracker,
    return_annotated=True
)

#counts = {"car": 20, "motorcycle": 6, "bus": 2, "truck": 3}
counts = Counter(counts_z2)
avg_time = {"car": 4.0, "motorcycle": 3.0, "bus": 5.5, "truck": 10.0}

#print(f"GST = {gst:.2f} s")  # 23.33 
##############################################################################################
def calculateTimeCycle(counts, avg_time, no_of_lanes):
    
    if not counts:
        return {"green": 10, "red": 10, "yellow": 5}

    numerator = sum(counts[c]*avg_time[c] for c in counts) +10

    gst = numerator / (no_of_lanes + 1)

    result = {"green": round(gst), "red": 10, "yellow": 5}

    return result

frameList = [
    {
    "frame": cv2.imread(SOURCE_IMG_PATH_1),
    "polygon" : polygon,
    "lanes": 2
    },
     {
         "frame": cv2.imread(SOURCE_IMG_PATH_2),
    "polygon" : polygon2,
    "lanes": 3
     },{
         "frame": cv2.imread(SOURCE_IMG_PATH_3),
    "polygon" : polygon3,
    "lanes": 2
     }
   ]

gst_values = []

for idx, item in enumerate(frameList):
    counts, annotated = count_classes_in_zone(
        item["frame"],
        item["polygon"],          # o zone2
        model,
        tracker,
        return_annotated=True
    )
    print(f"Frame {idx+1} - Counts: {counts}")
    gst = calculateTimeCycle(counts, avg_time,item["lanes"])
    value = {"cicle_times": gst, "frameAnnotated": annotated}
    gst_values.append(value)
    print(f"Frame {idx+1} - GST: {gst}")
    if annotated is not None:
        sv.plot_image(annotated)
        plt.savefig(f"resultado_frame_{idx+1}.png")

print (f"{gst_values}")
import streamlit as st

for idx, item in enumerate(gst_values):
    st.image(item["frameAnnotated"], caption=f"Frame {idx+1} - Cicle Times: {item['cicle_times']}")
##############################################################################################
# Detecto correctamente los 3 vehiculos
# import supervision as sv
# from ultralytics import YOLO
# import numpy as np
# import cv2

# image = cv2.imread(SOURCE_VIDEO_PATH2)
# model = YOLO("yolov8n.pt")
# tracker = sv.ByteTrack()

# polygon =np.array([[76, 687], [735, 708], [789, 286], [493, 283]])
# polygon_zone = sv.PolygonZone(polygon=polygon)

# result = model.predict(image)[0]
# detections = sv.Detections.from_ultralytics(result)
# detections = tracker.update_with_detections(detections)

# is_detections_in_zone = polygon_zone.trigger(detections)
# print(polygon_zone.current_count)


##############################################################################################
##############################################################################################
