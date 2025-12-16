# feature_extraction.py
import numpy as np
from scipy.signal import find_peaks
import mediapipe as mp

from utils import calculate_angle

mp_pose = mp.solutions.pose

def extract_gait_features(landmark_data, fps):
    """
    Extracts spatiotemporal and kinematic gait features from landmark data.
    """
    if not landmark_data:
        # Return empty values for all three outputs
        return {}, [], []

    # --- Spatiotemporal Features ---
    heel_y_coords = np.array([frame['left_heel'][1] for frame in landmark_data])
    heel_strike_frames, _ = find_peaks(-heel_y_coords, distance=int(fps * 0.4))
    
    gait_features = {}
    if len(heel_strike_frames) >= 2:
        duration_seconds = len(landmark_data) / fps
        num_steps = len(heel_strike_frames)
        cadence = (num_steps / duration_seconds) * 60
        gait_features['cadence_spm'] = cadence
        gait_features['step_count'] = num_steps

        step_lengths_pixels = []
        for i in range(len(heel_strike_frames) - 1):
            frame1_idx = heel_strike_frames[i]
            frame2_idx = heel_strike_frames[i + 1]
            heel_pos1_x = landmark_data[frame1_idx]['left_heel'][0]
            heel_pos2_x = landmark_data[frame2_idx]['left_heel'][0]
            step_length = abs(heel_pos2_x - heel_pos1_x)
            step_lengths_pixels.append(step_length)
        avg_step_length = np.mean(step_lengths_pixels) if step_lengths_pixels else 0
        gait_features['avg_step_length_pixels'] = avg_step_length

    # --- Kinematic Features (Joint Angles) ---
    knee_angles = []
    for frame in landmark_data:
        landmarks = frame['landmarks']
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
        
        angle = calculate_angle(l_hip, l_knee, l_ankle)
        knee_angles.append(angle)

    gait_features['avg_knee_angle'] = np.mean(knee_angles) if knee_angles else 0
    gait_features['max_knee_flexion'] = np.min(knee_angles) if knee_angles else 0
    
    # This line needs to return three values to match the main script
    return gait_features, heel_y_coords, heel_strike_frames