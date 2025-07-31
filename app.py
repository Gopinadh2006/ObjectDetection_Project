import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import cvzone
import torch
import tempfile
import time
import os

# Automatically download yolov8s.pt if not present
model_path = "yolov8s.pt"
if not os.path.exists(model_path):
    from urllib.request import urlretrieve
    print("🔽 Downloading YOLOv8 model...")
    urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt", model_path)

model = YOLO(model_path)

# Configure Streamlit page
st.set_page_config(page_title="Live Object Detection", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    video {
        width: 100% !important;
        height: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📸 Live Object Detection and Tracking")

# Load YOLOv8
model = YOLO("yolov8s.pt")
if torch.cuda.is_available():
    model.to("cuda")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
classNames = model.names

# IOU function
def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

# Streamlit camera and frame rendering
frame_placeholder = st.empty()

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

stop = st.button("🟥 Stop Camera")

while cap.isOpened() and not stop:
    success, frame = cap.read()
    if not success:
        st.warning("Failed to capture from webcam.")
        break

    results = model.predict(source=frame, conf=0.5, iou=0.5)[0]
    detections, class_map = [], []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append([x1, y1, x2, y2, conf])
        class_map.append((cls, [x1, y1, x2, y2]))

    detections_np = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
    tracked_objects = tracker.update(detections_np)

    person_count = 0
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        w, h = x2 - x1, y2 - y1

        best_match = "Unknown"
        max_iou = 0
        for cls_id, det_box in class_map:
            iou = compute_iou([x1, y1, x2, y2], det_box)
            if iou > max_iou:
                max_iou = iou
                best_match = classNames[cls_id]

        if best_match.lower() == "person":
            person_count += 1

        label = f"{best_match} ID: {int(track_id)}"
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, t=2)
        cvzone.putTextRect(frame, label, (x1, y1 - 10), scale=0.7, thickness=1)

    # Person count display
    cv2.rectangle(frame, (10, 10), (260, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"People Detected: {person_count}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Show frame in Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    # Slight delay for FPS control
    time.sleep(0.01)

cap.release()
