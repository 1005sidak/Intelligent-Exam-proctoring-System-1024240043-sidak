import cv2
import mediapipe as mp
import numpy as np

class EyeGazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True
        )

    def get_gaze(self, frame):
        img_h, img_w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return "NO FACE"

        for face_landmarks in results.multi_face_landmarks:
            # Left eye landmarks
            left_eye = [33, 133]
            right_eye = [362, 263]

            lx1 = int(face_landmarks.landmark[left_eye[0]].x * img_w)
            lx2 = int(face_landmarks.landmark[left_eye[1]].x * img_w)

            rx1 = int(face_landmarks.landmark[right_eye[0]].x * img_w)
            rx2 = int(face_landmarks.landmark[right_eye[1]].x * img_w)

            # Approximate center
            left_center = (lx1 + lx2) // 2
            right_center = (rx1 + rx2) // 2

            face_center = (left_center + right_center) // 2

            if face_center < img_w * 0.4:
                return "LEFT"
            elif face_center > img_w * 0.6:
                return "RIGHT"
            else:
                return "CENTER"