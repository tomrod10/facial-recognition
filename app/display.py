import os

import cv2
import processing

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SRC_DIR, "data/")


class Display:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.fr = processing.FacialRecognition()

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def start(self):
        self.fr.enroll_person("papi chulo gosling", os.path.join(DATA_PATH, "papi-chulo.jpg"))
        self.fr.enroll_person("legoat", os.path.join(DATA_PATH, "legoat.jpg"))

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Problem receiving frames...")
                break


            detected_faces = self.fr.process_live_frame(frame)
            self.fr.draw_overlays(frame, detected_faces)
            cv2.imshow("live feed", frame)

            if cv2.waitKey(1) == ord("q"):
                self.stop()
                break

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
