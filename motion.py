import cv2
import config


class MotionDetector:

    def __init__(self):

        self.previous = None

    def detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous is None:

            self.previous = gray

            return False

        diff = cv2.absdiff(self.previous, gray)

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

        self.previous = gray

        for contour in contours:

            if cv2.contourArea(contour) > config.MIN_CONTOUR_AREA:

                return True

        return False