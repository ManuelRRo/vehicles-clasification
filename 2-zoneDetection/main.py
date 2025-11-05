from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import threading
import torch
_thread_local = threading.local()
#Define polygons zones
# Polygons From PolygonZone
zones = [
   {
      'name': "Zone 1",
      'polygon': np.array([[908, 138], [1052, 138], [1267, 716], [1046, 733]]),
      'max': 15
   },   
   {
      'name': "Zone 2",
      'polygon': np.array([[18, 501], [115, 148], [225, 146], [115, 505]]),
      'max': 15
   },   
   {
      'name': "Zone 3",
      'polygon': np.array([[354, 679], [134, 686], [265, 138], [433, 138]]),
      'max': 15
   },   
   {
      'name': "Zone 4",
      'polygon': np.array([[646, 348], [461, 353], [470, 133], [642, 125]]),
      'max': 15
   },
   {
      'name': "Zone 5",
      'polygon': np.array([[698, 123], [848, 114], [889, 335],[698, 116]]),
      'max': 15
   }    
    # np.array([[908, 138], [1052, 138], [1267, 716], [1046, 733]]),
    # np.array([[18, 501], [115, 148], [225, 146], [115, 505]]),
    # np.array([[354, 679], [134, 686], [265, 138], [433, 138]])
    # np.array([[646, 348], [461, 353], [470, 133], [642, 125]]),
    # np.array([[698, 123], [848, 114], [889, 335],[698, 116]]),

]

zones2 = [
    {
        'name': "Zone 1",
        'polygon': np.array([[229, 50],[-3, 306],[1, 614],[369, 50]]),
        'max': 32
    },
    {
        'name': 'Zone 2',
        'polygon': np.array([[465, 46],[177, 574],[401, 578],[609, 46]]),
        'max': 38
    },
    {
        'name': 'Zone 3',
        'polygon': np.array([[697, 58],[461, 858],[737, 858],[849, 58]]),
        'max': 46
    },
    {
        'name': 'Zone 4',
        'polygon': np.array([[941, 58],[909, 862],[1273, 858],[1137, 58]]),
        'max': 48
    },
    {
        'name': 'Zone 5',
        'polygon': np.array([[1229, 46],[1501, 1078],[1889, 1078],[1405, 46]]),
        'max': 52
    }
]
model = YOLO("yolov8n.pt")

def callback(x: np.ndarray) -> sv.Detections:
    if not hasattr(_thread_local, "model"):
        _thread_local.model = YOLO("yolov8n.pt")  # modelo propio del hilo
    with torch.no_grad():
        results = _thread_local.model(x, conf=0.25, verbose=False)
    return sv.Detections.from_ultralytics(results[0])

tracker = sv.ByteTrack()
slicer = sv.InferenceSlicer(
    callback=callback,
    slice_wh=(800, 800),
    overlap_ratio_wh=None,     # deprecado -> dejar en None
    overlap_wh=(160, 160),     # 20% de 800
    thread_workers=0,          # <-- clave para evitar la condición de carrera
    iou_threshold=0.2
)

triangle_annotator = sv.TriangleAnnotator(base=20, height=20)
heat_map_annotator = sv.HeatMapAnnotator()

def setup_zones(frame_wh):
    if zones:
        for zone in zones:
            zone['history'] = []
            zone['PolygonZone'] = sv.PolygonZone(
                polygon=zone['polygon']   # sin frame_resolution_wh
            )
            zone['PolygonZoneAnnotator'] = sv.PolygonZoneAnnotator(
                zone=zone['PolygonZone'],
                color=sv.Color.WHITE,
                thickness=4,
            )

def process_frame(frame, heatmap=None):
    detections = slicer(image=frame)
    detections = tracker.update_with_detections(detections)

    annotated_frame = frame.copy()
    annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=detections)

    if heatmap is None:
        heatmap = np.full(frame.shape, 255, dtype=np.uint8)

    heat_map_annotator.annotate(scene=heatmap, detections=detections)

    if zones:
        for zone in zones:
            zone_presence = zone['PolygonZone'].trigger(detections)
            zone_present_idxs = [idx for idx, present in enumerate(zone_presence) if present]
            zone_present = detections[zone_present_idxs]
            zone_count = len(zone_present)
            zone['history'].append(zone_count)

            annotated_frame = zone['PolygonZoneAnnotator'].annotate(
                scene=annotated_frame, label=f"{zone['name']}: {zone_count}"
            )
            heatmap = zone['PolygonZoneAnnotator'].annotate(scene=heatmap, label=" ")

    return annotated_frame, heatmap

image = cv2.imread("./frames/bethoven1.png")
if image is None:
    raise FileNotFoundError("No se pudo abrir './frames/bethoven1.png'")

image_wh = (image.shape[1], image.shape[0])
setup_zones(image_wh)

annotated_image, heatmap = process_frame(image)

sv.plot_image(annotated_image)
sv.plot_image(heatmap)
plt.savefig("resultado.png")
# # Setup Model
# model = YOLO("yolov8n.pt")

# def callback(x: np.ndarray) -> sv.Detections:
#     result = model.predict(x, conf=0.25) 
#     return sv.Detections.from_inference(result)

# tracker = sv.ByteTrack()
# slicer = sv.InferenceSlicer(
#     callback=callback,
#     slice_wh=(800, 800),
#     overlap_ratio_wh=(0.2, 0.2),
#     thread_workers=10,
#     iou_threshold=0.2
# )
# triangle_annotator = sv.TriangleAnnotator(
#     base=20,
#     height=20
# )
# heat_map_annotator = sv.HeatMapAnnotator()

# def setup_zones(frame_wh):
#   if zones:
#     for zone in zones:
#       zone['history'] = []
#       zone['PolygonZone'] = sv.PolygonZone(
#           polygon=zone['polygon'],
#           #frame_resolution_wh=frame_wh
#       )
#       zone['PolygonZoneAnnotator'] = sv.PolygonZoneAnnotator(
#         zone=zone['PolygonZone'],
#         color=sv.Color.WHITE,
#         thickness=4,
#     )

# def process_frame(frame,heatmap=None):
#     detections = slicer(image=frame)
#     detections = tracker.update_with_detections(detections)

#     annotated_frame = frame.copy()

#     annotated_frame = triangle_annotator.annotate(
#         scene=annotated_frame,
#         detections=detections
#     )

#     if heatmap is None:
#       heatmap = np.full(frame.shape, 255, dtype=np.uint8)

#     heat_map_annotator.annotate(
#       scene=heatmap,
#       detections=detections
#     )

#     if zones:
#       for zone in zones:
#         zone_presence = zone['PolygonZone'].trigger(detections)
#         zone_present_idxs = [idx for idx, present in enumerate(zone_presence) if present]
#         zone_present = detections[zone_present_idxs]

#         zone_count = len(zone_present)
#         zone['history'].append(zone_count)


#         annotated_frame = zone['PolygonZoneAnnotator'].annotate(
#             scene=annotated_frame,
#             label=f"{zone['name']}: {zone_count}"
#         )

#         # Heatmap
#         heatmap = zone['PolygonZoneAnnotator'].annotate(
#             scene=heatmap,
#             label=" "
#         )

#     return annotated_frame, heatmap

# #Try with a single image
# image = cv2.imread("./frames/frame1.png")
# image_wh = (image.shape[1],image.shape[0])
# setup_zones(image_wh)

# annotated_image, heatmap = process_frame(image)

# sv.plot_image(annotated_image)
# sv.plot_image(heatmap)
# plt.savefig("resultado.png")