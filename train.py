from ultralytics import YOLO

def train_model():
    # Load pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train on PPE dataset (replace data.yaml path)
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="ppe_detection"
    )

if __name__ == "__main__":
    train_model()