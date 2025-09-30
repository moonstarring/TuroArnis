import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from ultralytics import YOLO
from .sort import Sort

class PoseAnalyzer:
    def __init__(self):
        print("[info] initializing computer vision components...")
        self.yolo_model = YOLO('yolov8n.pt')
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        try:
            self.classifier_model = joblib.load('models/arnis_random_forest_classifier.joblib')
            self.class_names = joblib.load('models/arnis_class_names.joblib')
            print("[info] classifier model loaded.")
        except FileNotFoundError:
            self.classifier_model = None
            self.class_names = None
            print("[error] classifier model not found. analysis will be limited to tracking.")
        print("[info] computer vision components ready.")

    def process_frame(self, frame, target_form=None):
        analysis_results = []
        
        results_yolo = self.yolo_model(frame, stream=True, verbose=False, classes=[0])
        detections = np.empty((0, 5))
        min_confidence = 0.5

        for r in results_yolo:
            for box in r.boxes:
                if box.conf[0] >= min_confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections = np.vstack((detections, np.array([x1, y1, x2, y2, box.conf[0]])))

        tracked_persons = self.tracker.update(detections)

        for person in tracked_persons:
            x1, y1, x2, y2, person_id = map(int, person)

            person_result = {
                'id': person_id,
                'bbox': (x1, y1, x2, y2),
                'predicted_class': None,
                'confidence': 0.0,
                'live_angles': None
            }
            
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                analysis_results.append(person_result)
                continue

            image_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(image_rgb)
            
            if pose_results.pose_landmarks:
                self._draw_landmarks_on_main_frame(frame, pose_results, x1, y1, x2, y2)
            
            if self.classifier_model and pose_results.pose_world_landmarks:
                try:
                    live_angles = self._calculate_all_angles_3d(pose_results.pose_world_landmarks.landmark)
                    person_result['live_angles'] = live_angles

                    if live_angles:
                        feature_order = [
                            'left_elbow', 'left_shoulder', 'left_hip', 'left_knee',
                            'right_elbow', 'right_shoulder', 'right_hip', 'right_knee'
                        ]

                        row = [live_angles.get(joint, 0) for joint in feature_order]

                        x_live = pd.DataFrame([row])
                        prediction_index = self.classifier_model.predict(x_live)[0]
                        predicted_class = self.class_names[prediction_index]
                        prediction_proba = self.classifier_model.predict_proba(x_live)[0]
                        confidence = prediction_proba[prediction_index]

                        person_result['predicted_class'] = predicted_class
                        person_result['confidence'] = confidence
                
                except Exception as e:
                    print(f"error during analysis for user {person_id}: {e}")

            analysis_results.append(person_result)

        return frame, analysis_results, tracked_persons

    def _calculate_angle_3d(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        dot_product = np.dot(ba, bc)
        magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
        cosine_angle = np.clip(dot_product / (magnitude + 1e-6), -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def _calculate_all_angles_3d(self, landmarks):
        try:
            coords = {lm.name.lower(): [lm.x, lm.y, lm.z] for lm in self.mp_pose.PoseLandmark}
            
            lm_data = {self.mp_pose.PoseLandmark(i).name.lower(): [landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in range(len(landmarks))}
            
            return {
                'left_elbow': self._calculate_angle_3d(lm_data['left_shoulder'], lm_data['left_elbow'], lm_data['left_wrist']),
                'left_shoulder': self._calculate_angle_3d(lm_data['left_hip'], lm_data['left_shoulder'], lm_data['left_elbow']),
                'left_hip': self._calculate_angle_3d(lm_data['left_shoulder'], lm_data['left_hip'], lm_data['left_knee']),
                'left_knee': self._calculate_angle_3d(lm_data['left_hip'], lm_data['left_knee'], lm_data['left_ankle']),
                'right_elbow': self._calculate_angle_3d(lm_data['right_shoulder'], lm_data['right_elbow'], lm_data['right_wrist']),
                'right_shoulder': self._calculate_angle_3d(lm_data['right_hip'], lm_data['right_shoulder'], lm_data['right_elbow']),
                'right_hip': self._calculate_angle_3d(lm_data['right_shoulder'], lm_data['right_hip'], lm_data['right_knee']),
                'right_knee': self._calculate_angle_3d(lm_data['right_hip'], lm_data['right_knee'], lm_data['right_ankle']),
            }
        except Exception:
            return None

    def _draw_landmarks_on_main_frame(self, main_frame, pose_results, x1, y1, x2, y2):
        # this helper method is now more robust
        if not pose_results.pose_landmarks:
            return

        frame_height, frame_width, _ = main_frame.shape
        crop_width = x2 - x1
        crop_height = y2 - y1

        # create a deep copy to avoid modifying the original landmarks
        # landmarks_copy = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        landmarks_copy = landmark_pb2.NormalizedLandmarkList()
        
        landmarks_copy.CopyFrom(pose_results.pose_landmarks)

        for landmark in landmarks_copy.landmark:
            pixel_x = landmark.x * crop_width + x1
            pixel_y = landmark.y * crop_height + y1
            
            landmark.x = pixel_x / frame_width
            landmark.y = pixel_y / frame_height

        self.mp_drawing.draw_landmarks(
            main_frame,
            landmarks_copy,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    def close(self):
        self.pose.close()
        print("[info] pose analyzer closed.")