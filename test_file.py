import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO('yolo11n-pose.pt')

# Open webcam
cap = cv2.VideoCapture(0)

# Recording control
recording = False
writer = None
output_file = "output.mp4"
output_path = "test_folder/"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# FPS measurement variables
prev_time = time.time()
smoothed_fps = 0

alpha = 0.1   # smoothing factor for FPS (0–1). Lower = smoother.

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- Measure FPS ---
    curr_time = time.time()
    instant_fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Smooth the FPS to avoid jitter
    smoothed_fps = (alpha * instant_fps) + (1 - alpha) * smoothed_fps

    # For debugging: print FPS every ~30 frames
    if int(curr_time) % 2 == 0:  
        print(f"Current FPS: {smoothed_fps:.2f}")

    # Pose estimation
    results = model(frame)
    annotated_frame = results[0].plot()

    # Write frame if recording
    if recording and writer is not None:
        writer.write(annotated_frame)

    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    # Start recording
    if key == ord('r') and not recording:
        print(f"➡ Recording started (FPS = {smoothed_fps:.2f})")

        height, width = annotated_frame.shape[:2]
        fps_to_use = max(1, min(int(smoothed_fps), 30))  # clamp between 1 and 30

        writer = cv2.VideoWriter(
            output_file,
            fourcc,
            fps_to_use,
            (width, height)
        )

        print(f"Recorded video FPS set to: {fps_to_use}")
        recording = True

    # Stop recording
    elif key == ord('s') and recording:
        print("⏹ Recording stopped")
        recording = False
        writer.release()
        writer = None

    # Quit
    elif key == ord('q'):
        break

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
