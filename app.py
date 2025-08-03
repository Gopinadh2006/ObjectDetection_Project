import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
from sort import Sort
import cv2
import numpy as np
import cvzone
import torch
import os

# =========================
# TURN/STUN CONFIGURATION
# =========================
RTCConfiguration={
    "iceServers": [
        {
            "urls": ["stun:stun.l.google.com:19302"]
        },
        {
            "urls": ["turn:0.tcp.in.ngrok.io:17515transport=tcp"],
            "username": "user",
            "credential": "pass"
        }
    ]
}

# =========================
# Load YOLOv8 Model
# =========================
model_path = "yolov8s.pt"
if not os.path.exists(model_path):
    from urllib.request import urlretrieve
    urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt", model_path)

model = YOLO(model_path)
if torch.cuda.is_available():
    model.to("cuda")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
classNames = model.names

# =========================
# IOU Calculator
# =========================
def compute_iou(box1, box2):
    xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

# =========================
# Video Processing
# =========================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.person_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(source=img, conf=0.5, iou=0.5)[0]

        detections, class_map = [], []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf])
            class_map.append((cls, [x1, y1, x2, y2]))

        detections_np = np.array(detections) if detections else np.empty((0, 5))
        tracked_objects = tracker.update(detections_np)

        self.person_count = 0
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
                self.person_count += 1

            label = f"{best_match} ID: {int(track_id)}"
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, t=2)
            cvzone.putTextRect(img, label, (x1, y1 - 10), scale=0.7, thickness=1)

        # Count Display
        cv2.rectangle(img, (10, 10), (270, 60), (0, 0, 0), -1)
        cv2.putText(img, f"People Detected: {self.person_count}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return img

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Live Object Detection", layout="wide")
st.title("📸 Live Object Detection and Tracking (YOLOv8 + SORT)")

webrtc_streamer(
    key="object-detect",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION
)


