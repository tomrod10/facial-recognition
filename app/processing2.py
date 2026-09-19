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

        # Look for the last dimension to map tensors accurately without strict shape crashes
        loc, conf = None, None
        for layer in outputs:
            layer_squeezed = np.squeeze(layer)
            if layer_squeezed.ndim >= 2:
                if layer_squeezed.shape[-1] == 4:
                    loc = layer_squeezed
                elif layer_squeezed.shape[-1] == 2:
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

    def enroll_static_profile(self, name, img_path):
        """
        Loads a static profile headshot, ensures a perfect square aspect ratio
        for accurate anchor mapping, and extracts its 512-D ArcFace vector.
        """
        import os

        if not os.path.exists(img_path):
            print(
                f"⚠️ Enrollment Warning: Could not find image file at '{img_path}'. Skipping."
            )
            return None

        static_img = cv2.imread(img_path)
        if static_img is None:
            print(
                f"⚠️ Enrollment Warning: File at '{img_path}' is corrupted or unreadable. Skipping."
            )
            return None

        # Force the image into a perfect square canvas to align anchors properly
        h, w, _ = static_img.shape
        max_side = max(h, w)
        square_canvas = np.zeros((max_side, max_side, 3), dtype=np.uint8)

        # Center the portrait inside the black square buffer canvas
        y_offset = (max_side - h) // 2
        x_offset = (max_side - w) // 2
        square_canvas[y_offset : y_offset + h, x_offset : x_offset + w] = static_img

        h, w, _ = square_canvas.shape

        # Build 640x640 input canvas tensor
        blob = cv2.dnn.blobFromImage(
            square_canvas,
            scalefactor=1.0,
            size=(640, 640),
            mean=(104.0, 117.0, 123.0),
            swapRB=False,
            crop=False,
        )
        self.detector.setInput(blob)

        out_names = self.detector.getUnconnectedOutLayersNames()
        outputs = self.detector.forward(out_names)

        # FIXED: Robust column index targeting
        loc, conf = None, None
        for layer in outputs:
            layer_squeezed = np.squeeze(layer)
            if layer_squeezed.ndim >= 2:
                if layer_squeezed.shape[-1] == 4:
                    loc = layer_squeezed
                elif layer_squeezed.shape[-1] == 2:
                    conf = layer_squeezed

        if loc is None or conf is None:
            print(
                f"❌ Enrollment Failed: Layer detection shape mismatch on '{img_path}'."
            )
            return None

        face_scores = conf[:, 1]
        valid_mask = face_scores > 0.75
        if not np.any(valid_mask):
            print(
                f"❌ Enrollment Failed: No clear faces detected in profile image '{img_path}'."
            )
            return None

        filtered_loc = loc[valid_mask]
        filtered_scores = face_scores[valid_mask]
        filtered_priors = self.priors[valid_mask]

        best_idx = np.argmax(filtered_scores)

        prior = filtered_priors[best_idx]
        edge = filtered_loc[best_idx]

        # Matrix box conversions
        cx = prior[0] + edge[0] * 0.1 * prior[2]
        cy = prior[1] + edge[1] * 0.1 * prior[3]
        box_w = prior[2] * np.exp(edge[2] * 0.2)
        box_h = prior[3] * np.exp(edge[3] * 0.2)

        real_x = int((cx - box_w / 2) * w)
        real_y = int((cy - box_h / 2) * h)
        real_w = int(box_w * w)
        real_h = int(box_h * h)

        x1, y1 = max(0, real_x), max(0, real_y)
        x2, y2 = min(w, real_x + real_w), min(h, real_y + real_h)

        if (x2 - x1) > 20 and (y2 - y1) > 20:
            face_roi = square_canvas[y1:y2, x1:x2]
            embedding = self._extract_arcface_embedding(face_roi)
            print(f" Successfully generated 512-D profile vector for '{name}'.")
            return embedding

        print(f"❌ Enrollment Failed: Bounding box region too small for '{img_path}'.")
        return None

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
