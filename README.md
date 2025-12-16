
# Live Gait Analysis Using MediaPipe

## Project Overview

This project performs **real-time gait analysis** using a webcam and **MediaPipe Pose**. It captures human gait in live video, extracts **spatiotemporal and kinematic features**, and visualizes gait events such as **heel strikes**. The system can be used for gait monitoring, rehabilitation tracking, or biomechanics research.

---

## Features

* **Live Webcam Capture**: Captures video from a webcam in real-time.
* **Pose Estimation**: Uses MediaPipe Pose to detect full-body landmarks.
* **Gait Feature Extraction**:

  * Cadence (steps per minute)
  * Step count
  * Average step length
  * Average knee angle
  * Maximum knee flexion
* **Heel Strike Detection**: Identifies heel strike frames using vertical heel position.
* **Visualization**: Plots heel Y-coordinate over time with detected heel strikes.
* **Modular Structure**: Easy-to-maintain modules (`feature_extraction.py`, `utils.py`, `visualization.py`).

---

## Folder Structure

```
LiveGaitAnalysis/
│
├── main.py                  # Entry point for running live gait analysis
├── feature_extraction.py    # Extracts gait features from landmark data
├── utils.py                 # Helper functions (e.g., angle calculation)
├── visualization.py         # Plotting functions for heel events
├── walking_video.mp4        # Optional video file for offline analysis
└── README.md                # Project documentation
```

---

## Requirements

Python 3.8+ and the following packages:

```bash
pip install opencv-python mediapipe matplotlib numpy scipy
```

---

## Usage

### 1. Run Live Gait Analysis via Webcam

```bash
python main.py
```

* The webcam opens and overlays pose landmarks.
* Walk in front of the camera for a few seconds.
* Press **Q** to stop recording.
* The system prints gait features and plots heel strike events.

### 2. Optional: Offline Video Analysis

* Place a video (e.g., `walking_video.mp4`) in the same folder.
* Modify `main.py` to read the video instead of webcam:

```python
cap = cv2.VideoCapture('walking_video.mp4')
```

---

## Modules Description

### `feature_extraction.py`

* Function: `extract_gait_features(landmark_data, fps)`
* Computes:

  * Spatiotemporal features: cadence, step length, step count
  * Kinematic features: knee angles, max flexion
  * Heel Y-coordinates & heel strike frames

### `utils.py`

* Function: `calculate_angle(a, b, c)`
* Calculates the angle (degrees) between three 2D points. Used for knee and other joint angles.

### `visualization.py`

* Function: `plot_gait_events(heel_y_coords, heel_strike_frames)`
* Plots heel vertical position with detected heel strikes.

### `main.py`

* Captures webcam feed.
* Extracts landmarks and passes them to `feature_extraction`.
* Prints gait features.
* Calls `plot_gait_events` to visualize results.


