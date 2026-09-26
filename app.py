import os

# Optimización de CPU para PyTorch y OpenMP
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import time
import cv2
import torch
from ultralytics import YOLO
from poses import classify_pose

torch.set_num_threads(2)

# Cargar modelo YOLOv8 Nano Pose
model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
prev_time = time.time()
annotated_frame = None
current_pose = "Detecting..."
current_angle = None

window_name = "Artificial Vision - Surveillance Stream"
cv2.namedWindow(window_name)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    frame_count += 1
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # Procesar inferencia YOLO cada 2 frames para ahorrar recursos
    if frame_count % 2 == 0:
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            current_pose, current_angle = classify_pose(keypoints)

    # Imagen a desplegar (si no hay cuadro anotado aún, usa el frame crudo)
    display_img = (
        annotated_frame if annotated_frame is not None else frame.copy()
    )

    # Superposición de FPS
    cv2.putText(
        display_img,
        f"FPS: {int(fps)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # Superposición del Ángulo de Cadera
    if current_angle is not None:
        cv2.putText(
            display_img,
            f"Hip Angle: {int(current_angle)} deg",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # Mapeo dinámico de colores según la postura detectada
    color_map = {
        "Sitting": (0, 255, 0),  # Verde
        "Crouching": (255, 165, 0),  # Naranja
        "Reaching": (255, 0, 255),  # Magenta
        "Standing / Other": (0, 165, 255),  # Amarillo / Ámbar
    }
    pose_color = color_map.get(
        current_pose, (255, 255, 255)
    )  # Blanco por defecto

    # Superposición de la Postura actual
    cv2.putText(
        display_img,
        f"Pose: {current_pose}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        pose_color,
        2,
    )

    cv2.imshow(window_name, display_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()