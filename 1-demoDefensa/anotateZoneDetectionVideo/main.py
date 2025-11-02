import numpy as np
import supervision as sv
from ultralytics import YOLO

SOURCE_VIDEO_PATH = "fuentesBethoven17seg.mp4"

POLYGON_ZONE1 = np.array([[954, 789], [960, 1074], [1085, 1071], [1070, 786]])

POLYGON_ZONE2 = np.array([[1132, 496], [1147, 674], [1770, 656], [1749, 463]])

POLYGON_ZONE3 = np.array([[933, 457], [829, 454], [832, 6], [927, 9]])



model = YOLO("traffic_analysis.pt")

tracker = sv.ByteTrack()

box_annotator = sv.RoundBoxAnnotator()

label_annotator = sv.LabelAnnotator()

trace_annotator = sv.TraceAnnotator()

zone = sv.PolygonZone(polygon=POLYGON_ZONE1)

zone2 = sv.PolygonZone(polygon=POLYGON_ZONE2)

zone3= sv.PolygonZone(polygon=POLYGON_ZONE3)

zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1)

zone_annotator2 = sv.PolygonZoneAnnotator(zone=zone2, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1)

zone_annotator3 = sv.PolygonZoneAnnotator(zone=zone3, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1)

video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    zone.trigger(detections=detections)
    zone2.trigger(detections=detections)
    zone3.trigger(detections=detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)
    
    annotated_frame = zone_annotator.annotate(scene=annotated_frame)
    
    annotated_frame = zone_annotator2.annotate(scene=annotated_frame)

    annotated_frame = zone_annotator3.annotate(scene=annotated_frame)
    
    return trace_annotator.annotate(
        annotated_frame, detections=detections)

sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path="result.mp4",
    callback=callback
)
