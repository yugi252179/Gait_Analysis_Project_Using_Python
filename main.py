# main.py
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

# Correct import statements for all local files
from feature_extraction import extract_gait_features
from visualization import plot_gait_events

def run_gait_analysis():
    """Main function to run the full gait analysis pipeline."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    landmark_data_buffer = []
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y]
            
            landmark_data_buffer.append({
                'landmarks': landmarks,
                'left_heel': left_heel
            })

            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Live Gait Analysis - Press Q to Quit', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n--- Final Gait Analysis Report ---")
    if landmark_data_buffer:
        gait_features, heel_y_coords, heel_strike_frames = extract_gait_features(landmark_data_buffer, fps)
        
        if gait_features:
            for key, value in gait_features.items():
                if isinstance(value, float):
                    print(f"{key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
        else:
            print("Not enough data collected for analysis.")
            
        plot_gait_events(heel_y_coords, heel_strike_frames)
            
    else:
        print("No poses were detected. Please ensure the person is in full view of the camera.")

if __name__ == '__main__':
    run_gait_analysis()