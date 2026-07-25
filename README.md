# AI Surveillance System using YOLOv8 and OpenCV

An AI-powered real-time surveillance system developed in Python using OpenCV and YOLOv8. The application detects human presence, identifies motion, captures snapshots, and records video automatically whenever a person is detected with motion.

## Features

- Real-time webcam surveillance
- Human detection using YOLOv8
- Motion detection using OpenCV
- Automatic snapshot capture
- Automatic video recording
- Date and time stamp on live video
- FPS (Frames Per Second) display
- People counter
- Motion status indicator
- REC recording indicator
- Manual snapshot using keyboard
- Fullscreen mode
- Automatic creation of Snapshots and Recordings folders

---

## Project Structure

```
AI-SURVEILLANCE-SYSTEM/
│
├── camera.py
├── config.py
├── detector.py
├── main.py
├── motion.py
├── recorder.py
├── utils.py
├── requirements.txt
├── README.md
│
├── Snapshots/
├── Recordings/
│
└── yolov8n.pt
```

---

## Technologies Used

- Python 3
- OpenCV
- Ultralytics YOLOv8
- NumPy

---

## Installation

### Clone Repository

```bash
git clone https://github.com/UmairAli-py/AI-Surveillance-System.git
```

```bash
cd AI-Surveillance-System
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download YOLOv8 Model

Download the YOLOv8 Nano model (`yolov8n.pt`) from the official Ultralytics release and place it inside the project folder.

The model file is not included in this repository.

---

## Run the Project

```bash
python main.py
```

---

## Keyboard Controls

| Key | Action |
|------|--------|
| Q | Quit Application |
| S | Manual Snapshot |
| F | Toggle Fullscreen |

---

## Output

### Snapshots

Captured images are automatically saved inside

```
Snapshots/
```

### Recordings

Recorded videos are automatically saved inside

```
Recordings/
```

---

## How It Works

1. Opens the webcam.
2. Continuously detects motion.
3. Runs YOLOv8 to detect humans.
4. If both motion and a person are detected:
   - Saves a snapshot.
   - Starts video recording.
5. Stops recording automatically after motion ends.
6. Displays live information including:
   - FPS
   - Motion status
   - People count
   - Date & Time
   - Recording indicator

---

## Future Improvements
- PyQt5 GUI
- IP Camera (RTSP) Support
- Multi-Camera Support
- Face Recognition
- Email Alerts
- Telegram Notifications
- Intruder Detection
- Database Logging
- PyQt5 GUI
- Recorded Video Analysis
- GPU (CUDA) Acceleration

---

## Known Limitations

- Detection accuracy decreases in low-light environments.
- Performance depends on webcam quality.
- The YOLOv8 Nano model prioritizes speed over maximum accuracy.
- Older webcams may produce noisy frames that reduce detection reliability.

---

## Requirements

- Python 3.10+
- Webcam
- Windows/Linux

---

## Author

**Umair Ali**

BS Computer Science Student

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- OpenCV
- Ultralytics YOLOv8
- Python Community
