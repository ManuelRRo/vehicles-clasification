from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import os
import cv2
HOME = os.getcwd()
SOURCE_VIDEO_PATH = f"{HOME}/proceres.mp4"
SOURCE_VIDEO_PATH2 = f"{HOME}/frames/frameProceres.png"

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
# initiate polygon zone
polygon = np.array([[76, 687], [735, 708], [789, 286], [493, 283]])


video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
zone = sv.PolygonZone(polygon=polygon)

# initiate annotators
box_annotator = sv.RoundBoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=1)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE, thickness=1, text_thickness=1, text_scale=1)

# extract video frame
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
iterator = iter(generator)
frame = next(iterator)
frame2 = image = cv2.imread(SOURCE_VIDEO_PATH2)
# detect
results = model(frame2)[0]
detections = sv.Detections.from_ultralytics(results)
detections = tracker.update_with_detections(detections)
zone.trigger(detections=detections)

# annotate
labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]
frame2 = box_annotator.annotate(scene=frame2, detections=detections)
frame2 = label_annotator.annotate(scene=frame2, detections=detections, labels=labels)
frame2 = zone_annotator.annotate(scene=frame2)
sv.plot_image(frame2)
plt.savefig("resultado.png")
##############################################################################################
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
