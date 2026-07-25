import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO

# ==========================
# Configuration
# ==========================

CAMERA_INDEX = 0

MODEL_PATH = "yolov8n.pt"

PERSON_CONFIDENCE = 0.50

MIN_CONTOUR_AREA = 1500

FPS_TARGET = 20

RECORD_TIMEOUT = 3

SNAPSHOT_FOLDER = "Snapshots"

VIDEO_FOLDER = "Recordings"

WINDOW_NAME = "AI Surveillance System"

FRAME_WIDTH = 1280

FRAME_HEIGHT = 720

YOLO_SKIP_FRAMES = 3

os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

os.makedirs(VIDEO_FOLDER, exist_ok=True)

# ==========================
# Load YOLO
# ==========================

print("Loading YOLOv8...")

model = YOLO(MODEL_PATH)

print("Model Loaded Successfully")

# ==========================
# Open Camera
# ==========================

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():

    raise Exception("Unable to open camera")

# ==========================
# Read First Frame
# ==========================

ret, previous_frame = cap.read()

if not ret:

    raise Exception("Unable to read first frame")

previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

previous_gray = cv2.GaussianBlur(previous_gray, (21,21), 0)

# ==========================
# Recording Variables
# ==========================

recording = False

video_writer = None

last_motion_time = 0

last_snapshot_time = 0

fullscreen = False

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

fps = 0

fps_time = time.time()

frame_count = 0

last_people = 0

last_boxes = []

# ==========================
# Helper Functions
# ==========================

def timestamp():

    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")


def image_name():

    return os.path.join(

        SNAPSHOT_FOLDER,

        datetime.now().strftime("%Y-%m-%d_%H-%M-%S.jpg")

    )


def video_name():

    return os.path.join(

        VIDEO_FOLDER,

        datetime.now().strftime("%Y-%m-%d_%H-%M-%S.mp4")

    )


def draw_info(frame, people, motion):

    cv2.putText(

        frame,

        timestamp(),

        (15,30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0,255,255),

        2

    )

    cv2.putText(

        frame,

        f"People : {people}",

        (15,60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0,255,0),

        2

    )

    cv2.putText(

        frame,

        f"FPS : {fps}",

        (15,90),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255,255,0),

        2

    )

    color = (0,255,0)

    text = "Motion : NO"

    if motion:

        color = (0,0,255)

        text = "Motion : YES"

    cv2.putText(

        frame,

        text,

        (15,120),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        color,

        2

    )

    if recording:

        cv2.circle(frame,(20,160),10,(0,0,255),-1)

        cv2.putText(

            frame,

            "REC",

            (40,168),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0,0,255),

            2

        )
# ==========================
# Main Loop
# ==========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ----------------------
    # FPS
    # ----------------------
    current_time = time.time()

    elapsed = current_time - fps_time

    if elapsed > 0:

        fps = int(1 / elapsed)

    fps_time = current_time

    # ----------------------
    # Motion Detection
    # ----------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    diff = cv2.absdiff(previous_gray, gray)

    thresh = cv2.threshold(
        diff,
        25,
        255,
        cv2.THRESH_BINARY
    )[1]

    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion = False

    for contour in contours:

        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:

            motion = True

            break

    previous_gray = gray.copy()

    # ----------------------
    # YOLO Human Detection (every Nth frame)
    # ----------------------
    frame_count += 1

    if frame_count % YOLO_SKIP_FRAMES == 0:

        people = 0

        current_boxes = []

        results = model.predict(
            frame,
            verbose=False
        )

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])

                conf = float(box.conf[0])

                if cls != 0:
                    continue

                if conf < PERSON_CONFIDENCE:
                    continue

                people += 1

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                current_boxes.append((x1, y1, x2, y2, conf))

        last_people = people

        last_boxes = current_boxes

    else:

        people = last_people

    for (x1, y1, x2, y2, conf) in last_boxes:

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Person {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ----------------------
    # Snapshot
    # ----------------------
    if motion and people > 0:

        if time.time() - last_snapshot_time > 5:

            filename = image_name()

            cv2.imwrite(filename, frame)

            print("Snapshot Saved:", filename)

            last_snapshot_time = time.time()

    # ----------------------
    # Recording Logic
    # ----------------------

    if motion and people > 0:

        last_motion_time = time.time()

        if not recording:

            filename = video_name()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                FPS_TARGET,
                (frame.shape[1], frame.shape[0])
            )

            recording = True

            print("Recording Started:", filename)

    if recording:

        if time.time() - last_motion_time > RECORD_TIMEOUT:

            recording = False

            video_writer.release()

            video_writer = None

            print("Recording Stopped")

    # ----------------------
    # Draw Information
    # ----------------------
    draw_info(
        frame,
        people,
        motion
    )

    # ----------------------
    # Write Frame To Video (after REC dot is drawn)
    # ----------------------
    if recording and video_writer is not None:

        video_writer.write(frame)

    cv2.imshow(
        WINDOW_NAME,
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Manual Snapshot
    if key == ord("s"):

        filename = image_name()

        cv2.imwrite(filename, frame)

        print("Manual Snapshot:", filename)

    # Fullscreen Toggle
    if key == ord("f"):

        fullscreen = not fullscreen

        if fullscreen:

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )

        else:

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
            )

    # Quit
    if key == ord("q"):

        break

# ==========================
# Cleanup
# ==========================

if video_writer is not None:

    video_writer.release()

cap.release()

cv2.destroyAllWindows()

print("Application Closed Successfully")