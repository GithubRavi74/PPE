from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")

def detect_image(image_path):
    results = model(image_path)
    results[0].show()
    results[0].save("output.jpg")

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("PPE Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_image("source_files/sample.jpg")