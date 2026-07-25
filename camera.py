import cv2
import config


class Camera:

    def __init__(self):

        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,
                     config.FRAME_WIDTH)

        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,
                     config.FRAME_HEIGHT)

    def read(self):

        return self.cap.read()

    def release(self):

        self.cap.release()