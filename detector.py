import cv2
from ultralytics import YOLO
import config


class HumanDetector:

    def __init__(self):

        print("Loading YOLO Model...")

        self.model = YOLO(config.MODEL_PATH)

        print("Model Loaded Successfully")

    def detect(self, frame):

        people = []

        results = self.model.predict(
            frame,
            verbose=False
        )

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])

                conf = float(box.conf[0])

                if cls == 0 and conf > config.PERSON_CONFIDENCE:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    people.append(

                        {
                            "box": (x1, y1, x2, y2),
                            "confidence": conf
                        }

                    )

        return people