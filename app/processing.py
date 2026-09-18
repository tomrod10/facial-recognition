import os

import cv2

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SRC_DIR, "yunet/face_detection_yunet_2023mar.onnx")


class FacialDetection:
    def __init__(self, model_path=MODEL_PATH, frame_size=(640, 480), score_threshold=0.6, nms_threshold=0.3):
        """
        Initializes the YuNet deep learning face detector.

        :param model_path: Path to the downloaded .onnx model weights file.
        :param frame_size: A tuple indicating (width, height) of the expected input frames.
        :param score_threshold: Confidence filter (0.0 to 1.0) for detecting faces.
        :param nms_threshold: Suppression threshold to prevent duplicate boxes for the same face.
        """
        self.model_path = model_path
        self.frame_size = frame_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        # Initialize the native OpenCV 5 face detector
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self.frame_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000
        )

    def update_frame_size(self, width, height):
        """Updates the input resolution dynamically if the video stream changes."""
        self.frame_size = (width, height)
        self.detector.setInputSize(self.frame_size)

    def detect_faces(self, frame):
        """ Processes a BGR image frame and extracts face parameters.

        :param frame: The raw BGR frame from your webcam feed.
        :return: A list of dicts, each containing 'box' (x, y, w, h), 'confidence', and 'landmarks'.
        """
        h, w = frame.shape[:2]
        if (w, h) != self.frame_size:
            self.update_frame_size(w, h)

        retval, faces = self.detector.detect(frame)

        results = []
        if faces is not None:
            for face in faces:
                results.append({
                    "box": face[0:4].astype(int),        # [x, y, width, height]
                    "landmarks": face[4:14].reshape(5, 2).astype(int),  # 5 keypoints: eyes, nose, mouth corners
                    "confidence": float(face[14])         # Confidence score
                })
        return results

    @staticmethod
    def draw_faces(frame, face_data, draw_landmarks=True):
        """
        Draws bounding boxes, confidence ratings, and key landmarks on the image.

        :param frame: The frame image to draw on.
        :param face_data: The list of dict items returned from detect_faces().
        :param draw_landmarks: Boolean toggle to show or hide facial feature dots.
        """
        for face in face_data:
            x, y, w, h = face["box"]
            confidence = face["confidence"]

            # Draw primary bounding rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Overlay confidence score label
            label = f"Face: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw facial landmarks (Eyes, Nose, Mouth Corners) if requested
            if draw_landmarks:
                for idx, point in enumerate(face["landmarks"]):
                    # Alternate colors for specific parts if you'd like (e.g. blue for eyes, red for nose)
                    cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)
