import os
HOME = os.getcwd()
print(HOME)
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from ultralytics import YOLO
from supervision.assets import download_assets, VideoAssets

#SOURCE_VIDEO_PATH = f"{HOME}/data/vehicles.mp4"
SOURCE_VIDEO_PATH = f"{HOME}/../sampleVideos/proceres-11-seg.mp4"

byte_tracker = sv.ByteTrack()
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(generator)

videoInfo = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print(videoInfo)
START = sv.Point(0, 280)
END = sv.Point(640, 280)

model = YOLO("yolo11x.pt")

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

bounding_box_annotator = sv.BoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.7)
trace_annotator = sv.TraceAnnotator(thickness=1)

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
    
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels)

    line_zone.trigger(detections)

    #crossed_in, crossed_out = line_zone.trigger(detections)
    detections_in = line_zone.out_count
    detections_out = line_zone.in_count
    print(detections_in)
    print(detections_out)

    return  line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
# sv.plot_image(annotated_frame, (12, 12))
# plt.savefig("resultado.png")
TARGET_VIDEO_PATH = f"{HOME}/count-objects-crossing-the-line-result.mp4"
sv.process_video(
    source_path = SOURCE_VIDEO_PATH,
    target_path = TARGET_VIDEO_PATH,
    callback=callback
)