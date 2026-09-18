import os

import cv2

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SRC_DIR, "yunet/")

class FacialRecognition:
    def __init__(self, src_dir=MODEL_PATH, frame_size=(640, 480)):
        # Paths to your src/ models
        self.detector_path = os.path.join(src_dir, "face_detection_yunet_2023mar.onnx")
        self.recognizer_path = os.path.join(src_dir, "face_recognition_sface_2021dec_int8bq.onnx")

        # 1. Initialize YuNet Detector
        self.detector = cv2.FaceDetectorYN.create(
            model=self.detector_path, config="", input_size=frame_size,
            score_threshold=0.6, nms_threshold=0.3
        )

        # 2. Initialize SFace Recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.recognizer_path, config=""
        )

        # Database structure to store {"Person Name": embedding_vector}
        self.known_faces = {}

        # OpenCV 5 SFace matching thresholds (Cosine Similarity metric)
        self.COSINE_THRESHOLD = 0.363
        self.NORML2_THRESHOLD = 1.128

    def enroll_person(self, name, image_path):
        """Extracts and saves a reference face embedding associated with a person's name."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return False

        # Temporarily configure detector size for the enrollment image
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        retval, faces = self.detector.detect(img)

        if faces is not None and len(faces) > 0:
            # Align and crop the face using YuNet's output details, then compute embedding
            aligned_face = self.recognizer.alignCrop(img, faces[0])
            embedding = self.recognizer.feature(aligned_face)
            self.known_faces[name] = embedding
            print(f"Successfully enrolled: '{name}'")
            return True
        else:
            print(f"Enrollment Failed: No face detected in profile image for {name}")
            return False

    def process_live_frame(self, frame):
        """Detects faces, computes embeddings, and matches them against the database."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        retval, faces = self.detector.detect(frame)

        recognized_people = []
        if faces is not None:
            for face in faces:
                box = face[0:4].astype(int)

                # Extract live embedding fingerprint
                aligned_face = self.recognizer.alignCrop(frame, face)
                live_embedding = self.recognizer.feature(aligned_face)

                # Match against database
                identity = "Unknown"
                highest_score = -1.0

                for name, known_embedding in self.known_faces.items():
                    # Calculate similarity score (1.0 means identical, lower means different)
                    score = self.recognizer.match(live_embedding, known_embedding, cv2.FaceRecognizerSF_FR_COSINE)

                    if score > self.COSINE_THRESHOLD and score > highest_score:
                        highest_score = score
                        identity = name

                recognized_people.append({
                    "box": box,
                    "name": identity,
                    "confidence": highest_score if identity != "Unknown" else float(face[14])
                })
        return recognized_people

    @staticmethod
    def draw_overlays(frame, identity_data):
        """Draws bounding boxes and labels names on the live camera viewport."""
        for person in identity_data:
            x, y, w, h = person["box"]
            name = person["name"]

            # Switch box colors: Green for recognized friends, Red for unknown faces
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{name}" if name == "Unknown" else f"{name} ({person['confidence']:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
