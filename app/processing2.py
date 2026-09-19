import os

import cv2
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RETINAFACE_PATH = os.path.join(SRC_DIR, "models/retinaface_mv2.onnx")
ARCFACE_PATH = os.path.join(SRC_DIR, "models/w600k_mbf.onnx")

class FaceRecognizerLight:
    def __init__(self, retinaface_path=RETINAFACE_PATH, arcface_path=ARCFACE_PATH):
        """Initializes RetinaFace-MobileNet and ArcFace-MobileFaceNet via cv2.dnn."""
        self.detector = cv2.dnn.readNetFromONNX(retinaface_path)
        self.recognizer = cv2.dnn.readNetFromONNX(arcface_path)

        # Optimize for Apple Silicon threads
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Pre-generate RetinaFace anchor grid for a 640x640 input to keep execution fast
        self.priors = self._generate_priors()

    def process_frame(self, frame, conf_threshold=0.8, nms_threshold=0.4):
        """Processes live frames using strict lightweight RetinaFace anchor mathematics."""
        h, w, _ = frame.shape

        # RetinaFace-MobileNet is highly optimized for a 640x640 canvas
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1.0, size=(640, 640),
            mean=(104.0, 117.0, 123.0), swapRB=False, crop=False
        )
        self.detector.setInput(blob)

        # RetinaFace outputs 3 layers: Bounding Box offsets, Classification confidence, and Landmarks
        out_names = self.detector.getUnconnectedOutLayersNames()
        loc, conf, _ = self.detector.forward(out_names)

        # Decode bounding boxes utilizing the anchor prior grid maps
        boxes, scores = self._decode_retinaface(loc, conf, w, h, conf_threshold)

        # Clear overlapping ghost boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)

        embeddings = []
        if len(indices) > 0:
            final_idx = indices.flatten() if hasattr(indices, 'flatten') else indices
            for i in final_idx:
                x, y, box_w, box_h = boxes[i]
                score = scores[i]

                # Draw clear tracking boxes on your UI window frame
                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {score:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Safe boundary box cropping
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + box_w), min(h, y + box_h)

                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    face_roi = frame[y1:y2, x1:x2]
                    embedding = self._extract_arcface_embedding(face_roi)
                    embeddings.append({"box": (x, y, box_w, box_h), "score": score, "embedding": embedding})

        return frame, embeddings

    def _generate_priors(self):
        """Generates standard RetinaFace multi-box anchor templates for a 640x640 canvas."""
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        feature_maps = [[80, 80], [40, 40], [20, 20]]

        anchors = []
        for k, f in enumerate(feature_maps):
            for i in range(f[0]):
                for j in range(f[1]):
                    for min_size in min_sizes[k]:
                        # Calculate center spatial positions normalized between 0 and 1
                        s_kx = j * steps[k] / 640.0
                        s_ky = i * steps[k] / 640.0
                        dense_cx = [s_kx]
                        dense_cy = [s_ky]
                        for cx, cy in zip(dense_cx, dense_cy):
                            anchors.append([cx, cy, min_size / 640.0, min_size / 640.0])

        return np.array(anchors, dtype=np.float32)

    def _decode_retinaface(self, loc, conf, frame_w, frame_h, threshold):
        """Decodes raw tensor offsets by combining them with pre-calculated anchors."""
        boxes, scores = [], []

        loc = np.squeeze(loc)   # Out Shape: (16800, 4)
        conf = np.squeeze(conf) # Out Shape: (16800, 2)

        # Softmax confidence parsing: index 1 represents target face presence probability
        face_scores = conf[:, 1] if conf.ndim > 1 else conf

        # Filter background array elements rapidly via vector lookups
        valid_indices = np.where(face_scores > threshold)[0]

        # Variational factors used to scale box variance distributions back to normal
        variances = [0.1, 0.2]

        for i in valid_indices:
            prior = self.priors[i]
            edge = loc[i]
            score = face_scores[i]

            # Mathematical anchor decoding formulas (Center-X, Center-Y, Width, Height)
            cx = prior[0] + edge[0] * variances[0] * prior[2]
            cy = prior[1] + edge[1] * variances[0] * prior[3]
            width = prior[2] * np.exp(edge[2] * variances[1])
            height = prior[3] * np.exp(edge[3] * variances[1])

            # Map percentages cleanly back to native pixel window space dimensions
            x1 = int((cx - width / 2) * frame_w)
            y1 = int((cy - height / 2) * frame_h)
            w_box = int(width * frame_w)
            h_box = int(height * frame_h)

            if w_box > 15 and h_box > 15:
                boxes.append([x1, y1, w_box, h_box])
                scores.append(float(score))

        return boxes, scores

    def _extract_arcface_embedding(self, face_roi):
        """Extracts deep identity embeddings natively using ArcFace MobileFaceNet."""
        aligned_face = cv2.resize(face_roi, (112, 112))

        # Dynamic normalization to scale integers down to [-1, 1] spatial dimensions
        face_blob = cv2.dnn.blobFromImage(
            aligned_face, scalefactor=1.0 / 127.5, size=(112, 112),
            mean=(127.5, 127.5, 127.5), swapRB=True
        )
        self.recognizer.setInput(face_blob)
        raw_embedding = self.recognizer.forward()

        # Unit-normalize vector via L2 norm
        norm = np.linalg.norm(raw_embedding)
        if norm > 0:
            raw_embedding /= norm

        return raw_embedding.flatten()
