# visualization.py
import matplotlib.pyplot as plt

def plot_gait_events(heel_y_coords, heel_strike_frames):
    """
    Plots the vertical movement of the heel and marks the detected heel strikes.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(heel_y_coords, label='Heel Vertical Position (Y-coordinate)')
    plt.plot(heel_strike_frames, heel_y_coords[heel_strike_frames], "x", color='red', markersize=10, label='Detected Heel Strikes')
    plt.title('Heel Position Analysis for Gait Event Detection')
    plt.xlabel('Frame Number')
    plt.ylabel('Normalized Y-coordinate (lower value is closer to ground)')
    plt.legend()
    plt.grid(True)
    plt.show()