🔍 Real-Time Object Detection & Tracking Web App



This project is a live object detection and tracking app built using YOLOv8, SORT tracker, and a Streamlit web interface. It allows you to detect and track objects (like people, cars, etc.) from your webcam in real-time, directly in your browser.

🚀 Features

🔎 Real-time object detection with YOLOv8

📌 Object tracking with SORT (Simple Online Realtime Tracking)

🎯 Class name & tracking ID display

👥 Real-time person counter

🌐 Clean, fullscreen Streamlit web UI (no extra controls)

💻 Runs locally on your webcam feed

🧠 Tech Stack

YOLOv8 - for object detection

SORT - for object tracking

OpenCV - for video capture and frame rendering

Streamlit - for lightweight browser UI

cvzone - for drawing utilities

🔧 Installation

Clone the repo

git clone https://github.com/your-username/object-detection.git
cd object-detection

Install dependencies

pip install -r requirements.txt

Or install manually:

pip install streamlit opencv-python-headless ultralytics cvzone numpy

Download YOLOv8 model

By default, this project uses yolov8s.pt

You can change to yolov8n.pt, yolov8m.pt, or yolov8l.pt in app.py

Start the app

streamlit run app.py

Visit the web app

http://localhost:8501

📁 Project Structure

object-detection/
│
├── app.py               # Streamlit app (main logic)
├── sort.py              # SORT tracker code
├── yolov8s.pt           # YOLOv8 model weights
├── requirements.txt     # Python dependencies
├── README.md            # Project overview
└── snapshots/           # Optional: saved frames if you add save feature

🎥 Preview

(Add a GIF or screenshot here of live detection if deploying to GitHub)

✅ Future Ideas



🧑‍💻 Author

Kota Hari Veera Sri Tarun

Managed by FRIDAY (AI Assistant)Feel free to connect via GitHub or LinkedIn.

🛡️ License

This project is licensed under the MIT License.

