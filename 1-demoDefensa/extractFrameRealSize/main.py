import cv2

VIDEO_PATH = "/home/manuel/repos/vehicles-clasification/sampleVideos/calleLosSisimiles-2-no-aerea.mp4"

def get_frame_native(video_path: str, frame_idx: int = None, time_s: float = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video")

    if frame_idx is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    elif time_s is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)

    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("No se pudo leer el frame solicitado")

    # frame_bgr es HxWx3 en tamaño nativo del video, sin ningún resize.
    return frame_bgr

# Ejemplo:
frame = get_frame_native(VIDEO_PATH, time_s=2)  # frame en el segundo 3.2, tamaño original
h, w = frame.shape[:2]
print("Tamaño nativo:", (w, h))  # Úsalo para definir PolygonZone
cv2.imwrite("frame_nativo2.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
