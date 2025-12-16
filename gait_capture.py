# gait_analyzer.py
#
# This script performs a basic gait analysis on a video of a person walking.
# It uses OpenCV for video processing and MediaPipe for pose estimation.
#
# Instructions:
# 1. Save this file as 'gait_analyzer.py'.
# 2. Place a video file named 'walking_video.mp4' in the same folder.
#    (The video should show a side-on view of a person walking).
# 3. Run the script from your terminal: python gait_analyzer.py

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def analyze_gait():
    """
    Main function to run the entire gait analysis process.
    It handles video input, landmark extraction, event detection, feature calculation,
    and result visualization.
    """
    # --- 1. INITIALIZATION ---
    # Initialize MediaPipe Pose solution
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # --- 2. VIDEO INPUT ---
    video_path = 'walking_video.mp4'  # The video file MUST be in the same folder
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        print("Please make sure the video file is in the same folder as the script and the filename is correct.")
        return

    # Get video properties like frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Use a default value if FPS is not available

    # --- 3. DATA COLLECTION (LANDMARK EXTRACTION) ---
    landmark_data = []
    frame_count = 0
    print("Processing video... Please wait.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            # End of video
            break

        # Convert the BGR image (OpenCV's default) to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Draw landmarks on the image for visualization
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract the coordinates of the left heel for analysis
            landmarks = results.pose_landmarks.landmark
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            
            # Store the data for the current frame
            landmark_data.append({
                'frame': frame_count,
                'left_heel': left_heel
            })
        
        # Display the video with pose landmarks
        cv2.imshow('Gait Analysis - Press ESC to Exit', image)
        frame_count += 1

        # Allow the user to exit by pressing the 'ESC' key
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Release video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Analyzed {frame_count} frames.")

    if not landmark_data:
        print("Error: No pose landmarks were detected in the video. The analysis cannot continue.")
        return

    # --- 4. GAIT EVENT DETECTION (HEEL STRIKES) ---
    # Extract the y-coordinate of the heel for each frame
    heel_y_coords = np.array([frame['left_heel'][1] for frame in landmark_data])
    
    # Find local minima (valleys) in the heel's y-coordinate trajectory.
    # A lower 'y' value on the screen means the heel is closer to the ground.
    # We find peaks in the *negative* y-array to identify these valleys.
    # 'height' and 'distance' parameters need to be tuned for different videos.
    heel_strike_frames, _ = find_peaks(-heel_y_coords, height=-0.95, distance=int(fps * 0.4))

    print(f"\nDetected {len(heel_strike_frames)} heel strikes at frames: {heel_strike_frames}")

    # --- 5. FEATURE EXTRACTION ---
    gait_features = {}
    if len(heel_strike_frames) >= 2:
        # Cadence (steps per minute)
        duration_seconds = frame_count / fps
        num_steps = len(heel_strike_frames)
        cadence = (num_steps / duration_seconds) * 60

        # Step Length (average horizontal distance between consecutive heel strikes)
        step_lengths_pixels = []
        for i in range(len(heel_strike_frames) - 1):
            frame1_idx = heel_strike_frames[i]
            frame2_idx = heel_strike_frames[i+1]
            
            # Get the horizontal position (x-coordinate) of the heel at each strike
            heel_pos1_x = landmark_data[frame1_idx]['left_heel'][0]
            heel_pos2_x = landmark_data[frame2_idx]['left_heel'][0]
            
            step_length = abs(heel_pos2_x - heel_pos1_x)
            step_lengths_pixels.append(step_length)

        avg_step_length = np.mean(step_lengths_pixels) if step_lengths_pixels else 0

        gait_features = {
            'cadence_spm': cadence,
            'avg_step_length_pixels': avg_step_length,
            'step_count': num_steps
        }
    
    # --- 6. FEEDBACK & RESULTS ---
    print("\n--- Gait Analysis Results ---")
    if gait_features:
        print(f"Cadence: {gait_features.get('cadence_spm', 0):.2f} steps per minute")
        print(f"Total Steps Detected: {gait_features.get('step_count', 0)}")
        print(f"Average Step Length: {gait_features.get('avg_step_length_pixels', 0):.3f} (in normalized screen width units)")
    else:
        print("Not enough heel strikes were detected to calculate gait features.")

    # --- 7. VISUALIZATION ---
    # Plot the vertical movement of the heel and mark the detected strikes
    plt.figure(figsize=(12, 6))
    plt.plot(heel_y_coords, label='Heel Vertical Position (Y-coordinate)')
    plt.plot(heel_strike_frames, heel_y_coords[heel_strike_frames], "x", color='red', markersize=10, label='Detected Heel Strikes')
    plt.title('Heel Position Analysis for Gait Event Detection')
    plt.xlabel('Frame Number')
    plt.ylabel('Normalized Y-coordinate (lower value is closer to ground)')
    plt.legend()
    plt.grid(True)
    plt.show()


# This is the "ignition switch" for the script.
# When the file is run directly, this block executes.
if __name__ == '__main__':
    analyze_gait()