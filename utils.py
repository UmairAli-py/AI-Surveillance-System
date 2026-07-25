import cv2
import time
from datetime import datetime


class FPS:

    def __init__(self):

        self.previous = time.time()
        self.fps = 0

    def update(self):

        current = time.time()

        self.fps = 1 / (current - self.previous)

        self.previous = current

        return int(self.fps)


def draw_datetime(frame):

    text = datetime.now().strftime("%d-%m-%Y  %H:%M:%S")

    cv2.putText(
        frame,
        text,
        (15,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )


def draw_fps(frame,fps):

    cv2.putText(
        frame,
        f"FPS : {fps}",
        (15,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )


def draw_people(frame,count):

    cv2.putText(
        frame,
        f"People : {count}",
        (15,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )


def draw_motion(frame,status):

    color=(0,255,0)

    text="Motion : NO"

    if status:

        color=(0,0,255)

        text="Motion : YES"

    cv2.putText(
        frame,
        text,
        (15,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def draw_recording(frame,status):

    if status:

        cv2.circle(frame,(20,160),10,(0,0,255),-1)

        cv2.putText(
            frame,
            "REC",
            (40,167),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )


def generate_filename(folder,extension):

    now=datetime.now()

    filename=now.strftime("%Y-%m-%d_%H-%M-%S")

    return f"{folder}/{filename}.{extension}"