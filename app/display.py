import os
import sys

import cv2
import processing2

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SRC_DIR, "data/")


class Display:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.fr = processing2.FacialRecognizer()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            print("Cannot open camera")
            sys.exit()

    def start(self):

        while True:
            ret, frame = self.cap.read()


            if not ret:
                print("Problem receiving frames...")
                break

            # processed_frame = cv2.flip(frame, 1)
            frame, emb = self.fr.process_frame(frame)
            cv2.imshow("live feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
