import os

import cv2
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RETINA_PATH = os.path.join(SRC_DIR, "models/retinaface-resnet50.onnx")
ARCFACE_PATH = os.path.join(SRC_DIR, "models/arcfaceresnet100-8.onnx")


class FacialRecognizer:
    def __init__(self, retinaface_path=RETINA_PATH, arcface_path=ARCFACE_PATH):
        """Initializes the heavyweight pipelines cleanly within your display context."""
        # Load networks via C++ backend (avoids framework memory overhead)
        self.detector = cv2.dnn.readNetFromONNX(retinaface_path)
        self.recognizer = cv2.dnn.readNetFromONNX(arcface_path)

        # Optimize execution targets for Apple Silicon CPU threads
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def process_frame(self, frame, conf_threshold=0.6, nms_threshold=0.4):
        """
        Accepts an incoming video frame.
        Detects faces, draws bounding boxes, and returns face embeddings.
        """
        h, w, _ = frame.shape

        # 1. RetinaFace Pre-processing (Expects 640x640 canvas and mean subtraction)
        # Note: If your display frames vary wildly, resize dynamically or pad to 640x640
        input_size = (640, 640)
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=input_size,
            mean=(104.0, 117.0, 123.0),
            swapRB=False,
            crop=False
        )
        self.detector.setInput(blob)

        # Forward pass pulls the raw regression anchors
        # Output 0: Bounding Box offsets, Output 1: Scores, Output 2: Landmark points
        out_names = self.detector.getUnconnectedOutLayersNames()
        loc, conf, landmarks = self.detector.forward(out_names)

        # 2. Decode RetinaFace Tensor Outputs
        boxes, scores, face_landmarks = self._decode_retinaface(loc, conf, landmarks, w, h, conf_threshold)

        # Apply Non-Maximum Suppression (NMS) to clear overlapping duplicate boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)

        embeddings = []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, box_w, box_h = boxes[i]
                score = scores[i]
                pts = face_landmarks[i]

                # Draw the bounding box onto the frame for your UI display
                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {score:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 3. Structural Crop & Identity Extraction
                # Crop face securely within boundary constraints
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + box_w), min(h, y + box_h)

                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    face_roi = frame[y1:y2, x1:x2]

                    # Compute ArcFace embedding vector
                    embedding = self._extract_arcface_embedding(face_roi)
                    embeddings.append({"box": (x, y, box_w, box_h), "embedding": embedding})

        return frame, embeddings

    def _extract_arcface_embedding(self, face_roi):
        """Processes localized face chips through the ResNet100 ArcFace pipeline."""
        # ArcFace strictly demands 112x112 pixel inputs normalized dynamically
        aligned_face = cv2.resize(face_roi, (112, 112))

        # Scales pixel integers from [0, 255] to [-1, 1] spatial dimensions
        face_blob = cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True
        )
        self.recognizer.setInput(face_blob)

        # Extract the deep 512-dimensional vector map
        raw_embedding = self.recognizer.forward()

        # Unit-normalize vector via L2 norm so you can check matches using np.dot()
        norm = np.linalg.norm(raw_embedding)
        if norm > 0:
            raw_embedding /= norm

        return raw_embedding.flatten()

    def _decode_retinaface(self, loc, conf, landmarks, frame_w, frame_h, threshold):
        """Decodes anchor tensors into real pixel coordinates mapping to the screen."""
        boxes, scores, face_landmarks = [], [], []

        # Squeeze down dimension footprints from forward arrays
        loc = np.squeeze(loc)
        conf = np.squeeze(conf)
        landmarks = np.squeeze(landmarks)

        # RetinaFace structures its scores where index 1 = face presence probability
        # For simplicity, this snippet handles standard unified bounding array transformations
        # (In production, replace this math layer with your anchor priors sheet if boxes drift)
        for i in range(len(conf)):
            score = conf[i][1] if conf.ndim > 1 else conf[i]
            if score < threshold:
                continue

            # Decode scaled coordinate translations matching bounding anchors
            # This turns relative percentage distributions into actual view coordinates
            box = loc[i]
            x_center = int(box[0] * frame_w)
            y_center = int(box[1] * frame_h)
            box_width = int(box[2] * frame_w)
            box_height = int(box[3] * frame_h)

            x = int(x_center - box_width / 2)
            y = int(y_center - box_height / 2)

            boxes.append([x, y, box_width, box_height])
            scores.append(float(score))

            # Map out landmark coordinate locations if present
            if len(landmarks) > 0:
                face_landmarks.append(landmarks[i])
            else:
                face_landmarks.append([])

        return boxes, scores, face_landmarks
