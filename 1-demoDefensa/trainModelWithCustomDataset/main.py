from ultralytics import YOLO
import torch

data_yaml = "/path/data.yaml" # path to your data.yaml
weights = "best-three.pt"     # start from a pretrained model (n = nano, s/m/l/x are larger)
epochs = 50
imgsz = 640
batch = 16
project = "runs_train"        # directory to save results
name = "yolov8n_custom"       # save results to project/name try with Yolo11npt
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Make sure paths exist
# ---- LOAD MODEL ----
model = YOLO(weights)  # e.g., "yolov8n.pt"

# ---- TRAIN ----
results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        workers=4,            # adjust if you have issues on Windows
        pretrained=True,      # use pretrained backbone
        exist_ok=True
    )
