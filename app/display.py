import os
import sys

import cv2
import numpy as np
import processing2

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SRC_DIR, "data/")


class Display:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.fr = processing2.FaceRecognizerLight()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame_count = 0
        self.last_embedding = []
        self.last_boxes = []
        self.user_database = {}

        if not self.cap.isOpened():
            print("Cannot open camera")
            sys.exit()

    def start(self):
        papi_vector = self.fr.enroll_static_profile(
            "papi-chulo-gosling", os.path.join(DATA_PATH, "papi-chulo.jpg")
        )
        legoat_vector = self.fr.enroll_static_profile(
            "Legoat", os.path.join(DATA_PATH, "legoat.jpg")
        )

        self.user_database["papi-chulo-gosling"] = papi_vector
        self.user_database["legoat"] = legoat_vector

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Problem receiving frames...")
                break
            frame, detected_faces = self.fr.process_frame(frame)

            for face in detected_faces:
                box = face["box"]
                embedding = face["embedding"]

                # --- RECOGNITION DATABASE LOOKUP ---
                identity = "Unknown"
                highest_score = 0.0

                for name, saved_embedding in self.user_database.items():
                    similarity = np.dot(embedding, saved_embedding)

                    if similarity > highest_score:
                        highest_score = similarity
                        if (
                            similarity > 0.45
                        ):  # Consistent ArcFace boundary verification filter
                            identity = name

                # Render identification banners
                x, y, bw, bh = box
                label = (
                    f"{identity} ({highest_score:.2f})"
                    if identity != "Unknown"
                    else "Unknown"
                )
                color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(
                    frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

            cv2.imshow("live feed", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exited program")
                break

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
