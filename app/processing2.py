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
        # self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # self.recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # self.recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Pre-generate the exact 16,800 anchor templates required by the mv2 backbone
        self.priors = self._generate_priors()

    def process_frame(self, frame, conf_threshold=0.75, nms_threshold=0.3):
        """Processes live frames using vectorized anchor transformation math."""
        h, w, _ = frame.shape

        # Build strict 640x640 input canvas tensor
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=(640, 640),
            mean=(104.0, 117.0, 123.0),
            swapRB=False,
            crop=False,
        )
        self.detector.setInput(blob)

        # Forward pass output elements
        out_names = self.detector.getUnconnectedOutLayersNames()
        outputs = self.detector.forward(out_names)

        # Sort out layers based on shape (Handles different layer order layouts)
        loc, conf = None, None
        for layer in outputs:
            layer_squeezed = np.squeeze(layer)
            if layer_squeezed.ndim == 2:
                if layer_squeezed.shape[1] == 4:
                    loc = layer_squeezed
                elif layer_squeezed.shape[1] == 2:
                    conf = layer_squeezed

        if loc is None or conf is None:
            return frame, []

        # Parse target face presence score safely from the classification layer
        face_scores = conf[:, 1]

        # Generate boolean lookup mask matching target threshold boundaries
        valid_mask = face_scores > conf_threshold
        if not np.any(valid_mask):
            return frame, []

        # Apply masks to isolate only valid face hits
        filtered_loc = loc[valid_mask]
        filtered_priors = self.priors[valid_mask]
        filtered_scores = face_scores[valid_mask]

        # --- THE CORRECTED RETINAFACE BOX VECTOR MATH ---
        # The model outputs raw offsets; we must scale them by standard variance parameters
        # and anchor them to our pre-generated priority coordinate grid map.
        cx = filtered_priors[:, 0] + filtered_loc[:, 0] * 0.1 * filtered_priors[:, 2]
        cy = filtered_priors[:, 1] + filtered_loc[:, 1] * 0.1 * filtered_priors[:, 3]
        box_w = filtered_priors[:, 2] * np.exp(filtered_loc[:, 2] * 0.2)
        box_h = filtered_priors[:, 3] * np.exp(filtered_loc[:, 3] * 0.2)

        boxes, scores = [], []
        for idx in range(len(filtered_scores)):
            # Convert normalized 0-1 percentages to native camera pixel dimensions
            real_w = int(box_w[idx] * w)
            real_h = int(box_h[idx] * h)
            real_x = int((cx[idx] - box_w[idx] / 2) * w)
            real_y = int((cy[idx] - box_h[idx] / 2) * h)

            if real_w > 20 and real_h > 20:
                boxes.append([real_x, real_y, real_w, real_h])
                scores.append(float(filtered_scores[idx]))

        # Clear duplicate bounding boxes overlapping the target area
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)

        embeddings = []
        if len(indices) > 0:
            final_idx = indices.flatten() if hasattr(indices, "flatten") else indices
            for i in final_idx:
                x, y, bw, bh = boxes[i]
                current_score = scores[i]

                # Render clear user visualization vectors onto screen display
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Face: {current_score:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # Target extraction boundary security clipping
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + bw), min(h, y + bh)

                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    face_roi = frame[y1:y2, x1:x2]
                    embedding = self._extract_arcface_embedding(face_roi)
                    embeddings.append(
                        {
                            "box": (x, y, bw, bh),
                            "score": current_score,
                            "embedding": embedding,
                        }
                    )

        return frame, embeddings

    def _generate_priors(self):
        """Generates standard RetinaFace multi-box anchor templates for a 640x640 canvas."""
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        feature_maps = [80, 40, 20]

        anchors = []
        for k, f in enumerate(feature_maps):
            for i in range(f):
                for j in range(f):
                    for min_size in min_sizes[k]:
                        s_kx = j * steps[k] / 640.0
                        s_ky = i * steps[k] / 640.0
                        anchors.append([s_kx, s_ky, min_size / 640.0, min_size / 640.0])

        return np.array(anchors, dtype=np.float32)

    def _extract_arcface_embedding(self, face_roi):
        """Extracts highly accurate vectors using w600k_mbf.onnx."""
        aligned_face = cv2.resize(face_roi, (112, 112))

        # Scale pixels down to [-1, 1] as required by ArcFace backbones
        face_blob = cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        self.recognizer.setInput(face_blob)
        raw_embedding = self.recognizer.forward()

        norm = np.linalg.norm(raw_embedding)
        if norm > 0:
            raw_embedding /= norm

        return raw_embedding.flatten()
