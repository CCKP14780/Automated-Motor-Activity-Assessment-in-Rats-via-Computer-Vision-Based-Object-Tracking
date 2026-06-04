import pandas as pd
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from pprint import pprint

# import gridDetection as grid
# Parameters from gridDetection.py --OBSOLETE
# DIMENSION = grid.get_grid_detection(display=False)  # (ROW, COLUMN, x, y, w, h)
# ROW = DIMENSION[0]
# COLUMN = DIMENSION[1]
# X_ARENA = DIMENSION[2]
# Y_ARENA = DIMENSION[3]
# W_ARENA = DIMENSION[4]
# H_ARENA = DIMENSION[5]

# Tracking parameters (CSV)
# TODO: INPUT_CSV = r'predictions\predictions\baseLine_labels.v003.000_mice_new.analysis.csv'
INPUT_CSV = r'Result\new_video.v002.000_mice_new.analysis.csv'
BODY_PART = 'torso'
X_COL, Y_COL = f'{BODY_PART}.x', f'{BODY_PART}.y'
DATA = pd.read_csv(INPUT_CSV)

# define arena based on data range + padding
# assuming the rat's thigmotaxis(wall-hugging) behavior, we can ensure that the area of the OFT arena can be defined with a small padding around the min/max coordinates of the tracked points. This allows us to capture all movements while ignoring outliers or tracking errors that may occur outside the arena boundaries.
ROW = 3
COLUMN = 5

padding = 5 

# base calculation
x_min_data = DATA[X_COL].min()
x_max_data = DATA[X_COL].max()
y_min_data = DATA[Y_COL].min()
y_max_data = DATA[Y_COL].max()

# sided tweaking
left_extension = 0.1 * (x_max_data - x_min_data)  # extend left side by 10% of the width

X_ARENA = (x_min_data - padding) - left_extension # left wall
Y_ARENA = y_min_data - padding # top wall (Y increases downwards in image coordinates)

W_ARENA = (x_max_data - x_min_data) + (2 * padding) + left_extension
H_ARENA = (y_max_data - y_min_data) + (2 * padding)
# print(f"NEW Arena Definition: X={X_ARENA:.1f}, Y={Y_ARENA:.1f}, W={W_ARENA:.1f}, H={H_ARENA:.1f}")

# --- SANITY CHECK ---
# print(f"Arena Definition: X={X_ARENA}, Y={Y_ARENA}, W={W_ARENA}, H={H_ARENA}")
# print(f"Track Data Range: X=[{DATA[X_COL].min():.1f}, {DATA[X_COL].max():.1f}], Y=[{DATA[Y_COL].min():.1f}, {DATA[Y_COL].max():.1f}]")

if DATA[X_COL].max() > (X_ARENA + W_ARENA):
    print("WARNING: Mouse moves OUTSIDE the defined arena width! Steps will be missed.")

def get_grid_index(X, Y):
    # Check if position is valid
    if not is_valid_position(X, Y):
        return None

    # relative position from the arena start
    rel_x = X - X_ARENA
    rel_y = Y - Y_ARENA

    # index (pos/cell)
    col = int(rel_x / (W_ARENA / COLUMN))
    row = int(rel_y / (H_ARENA / ROW))

    # clamp the values
    # 5 box becomes [0-4]
    col = min(col, COLUMN - 1)
    row = min(row, ROW - 1)
    
    # safety clamp for negative values
    col = max(col, 0)
    row = max(row, 0)

    return (row, col)

# grid validity check
def is_valid_grid(grid):
    return (
        grid is not None
        and isinstance(grid, tuple)
        and len(grid) == 2
        and grid[0] is not None
        and grid[1] is not None
    )

def is_valid_position(X, Y):
    return (
        X_ARENA <= X <= X_ARENA + W_ARENA
        and Y_ARENA <= Y <= Y_ARENA + H_ARENA
        and not np.isnan(X)
        and not np.isnan(Y)
    )

def calculate_grid_crossing(dwell_enter=5, dwell_return=30, show_changes=False):
    """
    dwell_enter:  Frames required to commit to a NEW grid (fast, for running).
    dwell_return: Frames required to return to the IMMEDIATE PREVIOUS grid (slow, for jitter).
    show_changes: If True, prints out every grid change with step count.
    """
    total_steps = {}
    temp_frame = 0
    for track_id in DATA['track'].unique():
        track_data = DATA[DATA['track'] == track_id]
        
        step_count = 0
        
        current_grid = None       # The grid the instance is officially in
        previous_grid = None      # The grid the instance was in before 'current'
        
        potential_grid = None     # The grid the instance is currently looking at
        frames_in_potential = 0
        
        for _, row in track_data.iterrows():
            X, Y = row[X_COL], row[Y_COL]
            
            # Get grid index
            grid = get_grid_index(X, Y)
            
            if grid is None:
                continue 
                
            # init on first frame
            if current_grid is None:
                current_grid = grid
                continue
            
            # COUNTING
            # if stable. Reset potential.
            if grid == current_grid:
                frames_in_potential = 0
                potential_grid = None
                
            else:
                # instance moving to a different grid.
                if grid == potential_grid:
                    frames_in_potential += 1
                else:
                    # Switched to a brand new potential grid
                    potential_grid = grid
                    frames_in_potential = 1
                
                # JITTERING THRESHOLD
                # if the instance are going back to the same place -anticipating jitter - be strict
                # if the instance are going to new place -running - be loose
                threshold = dwell_return if (grid == previous_grid) else dwell_enter
                
                if frames_in_potential >= threshold:
                    step_count += 1
                    
                    if show_changes:
                        move_type = "RETURN" if grid == previous_grid else "NEW"
                        print(f'{track_id}: {current_grid} -> {grid} ({move_type}) | Steps: {step_count}')
                    
                    # update State
                    previous_grid = current_grid
                    current_grid = grid
                    frames_in_potential = 0
                    potential_grid = None
                    
        total_steps[track_id] = step_count
        print(f'Processed {track_id}: {step_count} steps')
        print(f'Last frame for {track_id}: {row["frame_idx"]}')
        
    return total_steps

def calculate_instances_velocity():
    data = pd.read_csv(INPUT_CSV)
    velocities = {}

    for track_id in data['track'].unique():
        track_data = data[data['track'] == track_id]

        prev_x, prev_y = None, None
        frame_distances = []

        for _, row in track_data.iterrows():
            x, y = row[X_COL], row[Y_COL]

            if not is_valid_position(x, y):
                prev_x, prev_y = None, None
                continue

            if prev_x is not None:
                dx = x - prev_x
                dy = y - prev_y
                dist = np.sqrt(dx*dx + dy*dy)
                frame_distances.append(dist)

            prev_x, prev_y = x, y

        velocities[track_id] = np.mean(frame_distances) if frame_distances else 0

    return velocities

def run():
    # eg. dwell_enter=5  (approx 0.15s) captures fast running.
    # eg. dwell_return=30 (approx 1.0s) ignores jittering back and forth.
    step_counts = calculate_grid_crossing(dwell_enter=1, dwell_return=3, show_changes=True)
    print("TOTAL STEPS PER MOUSE:")
    for track, steps in step_counts.items():
        print(f'{track}: {steps}')

    velocity_dict = calculate_instances_velocity()
    print("\nAVERAGE VELOCITY PER MOUSE:")
    for track, velocity in velocity_dict.items():
        print(f'{track}: {velocity:.2f} pixels/frame')


run()