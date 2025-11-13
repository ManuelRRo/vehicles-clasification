# archivo recuperado de la m[aquinas ] virtual con gpu
import os
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from ultralytics import YOLO
from supervision.assets import download_assets, VideoAssets
from collections import defaultdict

HOME = os.getcwd()
SOURCE_VIDEO_PATH = f"../../../resources/videos/proceres.mp4"
SOURCE_MODEL_PATH = f"../../../resources/models/yolo11x.pt"
byte_tracker = sv.ByteTrack()
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(generator)

videoInfo = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print(videoInfo)
START = sv.Point(0, 280)
END = sv.Point(640, 280)

#START = sv.Point(631, 207)
#END = sv.Point(3, 223)


model = YOLO(SOURCE_MODEL_PATH)

line_zone = sv.LineZone(start=START, end=END)

line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_thickness=4,
    text_scale=0.8)

results = model(frame, verbose=False)[0]

detections = sv.Detections.from_ultralytics(results)


labels = [
    f"{results.names[class_id]} {confidence:0.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

box_annotator = sv.BoxAnnotator(thickness=6)
bounding_box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
trace_annotator = sv.TraceAnnotator(thickness=4)
annotated_frame = frame.copy()
annotated_frame = box_annotator.annotate(annotated_frame, detections)
annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
annotated_frame = frame.copy()
annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
# Diccionario para acumular conteos por clase
cross_counts = defaultdict(int)

def callback(frame: np.ndarray, index:int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]

    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame,
        detections=detections)
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame,
        detections=detections)
    # annotated_frame = label_annotator.annotate(
    #     scene=annotated_frame,
    #     detections=detections,
    #     labels=labels)


# --- dentro de tu loop de procesamiento ---
    crossed_in, crossed_out = line_zone.trigger(detections)
#    detections_in, detections_out = line_zone.trigger(detections)
    detections_out = detections[crossed_out]
    detections_in = detections[crossed_in]

    class_id_test = detections_out.class_id

    if len(class_id_test) > 0:
        for cls_id in class_id_test:
            cls_id = int(cls_id)
            # nombre de la clase según el modelo
            class_name = model.model.names[cls_id]

            # acumular conteo
            cross_counts[class_name] += 1

            # imprimir detección en tiempo real
            print(f"Detección out: class {cls_id} ({class_name})")

    # también puedes imprimir el resumen acumulado
    print("Resumen acumulado de cruces:", dict(cross_counts))

    return  line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
# sv.plot_image(annotated_frame, (12, 12))
# plt.savefig("resultado.png")
TARGET_VIDEO_PATH = f"{HOME}/count-objects-crossing-the-line-result.mp4"
sv.process_video(
    source_path = SOURCE_VIDEO_PATH,
    target_path = TARGET_VIDEO_PATH,
    callback=callback
)
