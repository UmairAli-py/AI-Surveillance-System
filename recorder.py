import cv2
import os
import time
from datetime import datetime

import config


class Recorder:

    def __init__(self):

        os.makedirs(config.VIDEO_FOLDER, exist_ok=True)
        os.makedirs(config.SNAPSHOT_FOLDER, exist_ok=True)

        self.writer = None
        self.recording = False
        self.last_motion_time = 0

    def _video_name(self):

        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.mp4")

        return os.path.join(config.VIDEO_FOLDER, name)

    def _image_name(self):

        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.jpg")

        return os.path.join(config.SNAPSHOT_FOLDER, name)

    def save_snapshot(self, frame):

        filename = self._image_name()

        cv2.imwrite(filename, frame)

        print(f"[Snapshot] {filename}")

    def start_recording(self, frame):

        if self.recording:
            return

        filename = self._video_name()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        height, width = frame.shape[:2]

        self.writer = cv2.VideoWriter(
            filename,
            fourcc,
            config.FPS,
            (width, height)
        )

        self.recording = True

        self.last_motion_time = time.time()

        print(f"[Recording Started] {filename}")

    def update(self, frame, motion):

        if not self.recording:
            return

        self.writer.write(frame)

        if motion:
            self.last_motion_time = time.time()

        if time.time() - self.last_motion_time > config.RECORD_AFTER_MOTION:

            self.stop_recording()

    def stop_recording(self):

        if not self.recording:
            return

        self.writer.release()

        self.writer = None

        self.recording = False

        print("[Recording Stopped]")

    def is_recording(self):

        return self.recording