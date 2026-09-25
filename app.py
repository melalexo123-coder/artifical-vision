import os

# Limit PyTorch CPU threads to keep CPU utilization low
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

torch.set_num_threads(2)

# Load the nano pose model (downloads automatically on first run ~6MB)
model = YOLO("yolov8n-pose.pt")


def calculate_angle(a, b, c):
    """Calculates 2D angle (in degrees) at vertex point b."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# Open default local webcam
cap = cv2.VideoCapture(0)

# Set lower resolution (640x480) to lighten CPU workload
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
prev_time = time.time()
annotated_frame = None

window_name = "YOLOv8 Pose - Local Stream"
cv2.namedWindow(window_name)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    frame_count += 1
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # Process YOLO inference on every 2nd frame to save CPU
    if frame_count % 2 == 0:
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        if (
            results[0].keypoints is not None
            and len(results[0].keypoints.xy) > 0
        ):
            keypoints = results[0].keypoints.xy[0].cpu().numpy()

            # Extract 5: L_Shoulder, 11: L_Hip, 13: L_Knee
            if len(keypoints) >= 14:
                l_shoulder, l_hip, l_knee = (
                    keypoints[5],
                    keypoints[11],
                    keypoints[13],
                )

                if (
                    np.any(l_shoulder)
                    and np.any(l_hip)
                    and np.any(l_knee)
                ):
                    hip_angle = calculate_angle(
                        l_shoulder, l_hip, l_knee
                    )
                    pose = (
                        "Sitting"
                        if 70 <= hip_angle <= 115
                        else "Standing / Other"
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Hip Angle: {int(hip_angle)} deg",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Pose: {pose}",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0) if pose == "Sitting" else (0, 165, 255),
                        2,
                    )

    # Show video stream
    display_img = (
        annotated_frame if annotated_frame is not None else frame
    )
    cv2.putText(
        display_img,
        f"FPS: {int(fps)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.imshow("YOLOv8 Pose - Local Stream", display_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()