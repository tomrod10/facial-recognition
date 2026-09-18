import cv2
import processing


class Display:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.fd = processing.FacialDetection()

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def start(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Problem receiving frames...")
                break

            detected_faces = self.fd.detect_faces(frame)
            self.fd.draw_faces(frame, detected_faces, True)
            cv2.imshow("live feed", frame)

            if cv2.waitKey(1) == ord("q"):
                self.stop()
                break

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
