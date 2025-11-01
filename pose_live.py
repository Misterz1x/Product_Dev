import cv2
from ultralytics import YOLO


# Load YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Open webcam video stream
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation
    results = model(frame)

    # Visualize results
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()