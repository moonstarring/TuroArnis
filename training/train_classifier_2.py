# file: train_classifier.py

# import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
# new: import the randomforestclassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# initialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def calculate_angle_3d(a, b, c):
    # calculates the angle between three 3d points
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot_product = np.dot(ba, bc)
    magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
    cosine_angle = np.clip(dot_product / (magnitude + 1e-6), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def extract_features_from_image(image_path):
    # processes a single image and returns its calculated angles
    image = cv2.imread(image_path)
    if image is None: 
        print(f"Failed to load image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_world_landmarks:
        print(f"No pose landmarks detected in: {image_path}")
        return None

    try:
        landmarks = results.pose_world_landmarks.landmark
        coords = {}
        
        # Add error checking for landmark extraction
        for lm in mp_pose.PoseLandmark:
            if not landmarks[lm.value]:
                print(f"Missing landmark {lm.name} in: {image_path}")
                return None
            coords[lm.name.lower()] = [
                landmarks[lm.value].x,
                landmarks[lm.value].y,
                landmarks[lm.value].z
            ]
        
        angles = {
            'left_elbow': calculate_angle_3d(coords['left_shoulder'], coords['left_elbow'], coords['left_wrist']),
            'left_shoulder': calculate_angle_3d(coords['left_hip'], coords['left_shoulder'], coords['left_elbow']),
            'left_hip': calculate_angle_3d(coords['left_shoulder'], coords['left_hip'], coords['left_knee']),
            'left_knee': calculate_angle_3d(coords['left_hip'], coords['left_knee'], coords['left_ankle']),
            'right_elbow': calculate_angle_3d(coords['right_shoulder'], coords['right_elbow'], coords['right_wrist']),
            'right_shoulder': calculate_angle_3d(coords['right_hip'], coords['right_shoulder'], coords['right_elbow']),
            'right_hip': calculate_angle_3d(coords['right_shoulder'], coords['right_hip'], coords['right_knee']),
            'right_knee': calculate_angle_3d(coords['right_hip'], coords['right_knee'], coords['right_ankle']),
        }
        return angles
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# --- main training pipeline ---
if __name__ == "__main__":
    dataset_folder = 'dataset_poses'
    csv_output_file = 'arnis_poses_for_classification.csv'
    
    # Check if dataset folder exists
    if not os.path.exists(dataset_folder):
        raise ValueError(f"Dataset folder not found: {dataset_folder}")
    
    pose_classes = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
    if not pose_classes:
        raise ValueError(f"No class folders found in {dataset_folder}")
    
    print(f"Found {len(pose_classes)} classes: {pose_classes}")
    
    total_images = 0
    processed_images = 0
    failed_images = 0
    
    header = ['class'] + [
        'left_elbow', 'left_shoulder', 'left_hip', 'left_knee',
        'right_elbow', 'right_shoulder', 'right_hip', 'right_knee'
    ]

    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for class_name in pose_classes:
            print(f"\nProcessing class: {class_name}")
            class_folder_path = os.path.join(dataset_folder, class_name)
            class_images = os.listdir(class_folder_path)
            total_images += len(class_images)
            
            class_processed = 0
            for filename in class_images:
                image_path = os.path.join(class_folder_path, filename)
                angles = extract_features_from_image(image_path)
                
                if angles:
                    row = [class_name] + [angles.get(joint, 0) for joint in header[1:]]
                    writer.writerow(row)
                    processed_images += 1
                    class_processed += 1
                else:
                    failed_images += 1
            
            print(f"Processed {class_processed}/{len(class_images)} images for class {class_name}")

    print(f"\n[Summary]")
    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Failed to process: {failed_images}")

    if processed_images == 0:
        raise ValueError("No images were successfully processed. Check the dataset and error messages above.")

    print("\n[info] starting random forest model training...")
    
    data = pd.read_csv(csv_output_file)
    X = data.drop('class', axis=1)
    y = data['class']

    # Print class distribution
    print("\nClass distribution:")
    class_counts = y.value_counts()
    print(class_counts)
    
    # Calculate minimum samples needed for stratification
    min_samples_needed = int(0.2 * len(y))  # 20% for test set
    if min_samples_needed < len(class_counts):
        print("\n[warning] Not enough samples for stratified split. Using simple random split instead.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # initialize the random forest classifier
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        # Add class balancing since we might have imbalanced data
        class_weight='balanced'
    )
    
    print("[info] training the model...")
    model.fit(X_train, y_train)

    # --- 3. evaluation ---
    print("[info] evaluating the model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nmodel accuracy: {accuracy * 100:.2f}%")
    print("\nclassification report:")
    print(classification_report(y_test, y_pred))

    # --- 4. saving the model ---
    model_filename = 'models/arnis_random_forest_classifier.joblib'
    class_names_filename = 'models/arnis_class_names.joblib'
    
    os.makedirs('models', exist_ok=True) # create models folder if it doesn't exist
    joblib.dump(model, model_filename)
    joblib.dump(model.classes_, class_names_filename)

    print(f"\n[info] training complete. model saved to {model_filename}")

    pose.close()