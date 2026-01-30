import gridDetection as grid
import pandas as pd
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from pprint import pprint

# Parameters from gridDetection.py
DIMENSION = grid.get_grid_detection(display=False)  # (ROW, COLUMN, x, y, w, h)
ROW = DIMENSION[0]
COLUMN = DIMENSION[1]
X_ARENA = DIMENSION[2]
Y_ARENA = DIMENSION[3]
W_ARENA = DIMENSION[4]
H_ARENA = DIMENSION[5]

# Tracking parameters (from CSV)
INPUT_CSV = r'predictions\predictions\baseLine_labels.v003.000_mice_new.analysis.csv'
BODY_PART = 'torso'
X_COL, Y_COL = f'{BODY_PART}.x', f'{BODY_PART}.y'
DATA = pd.read_csv(INPUT_CSV)

def calculate_grid_range():
    x_step = W_ARENA / COLUMN
    y_step = H_ARENA / ROW

    x_ranges = [(X_ARENA + i * x_step, X_ARENA + (i + 1) * x_step) for i in range(COLUMN)]
    y_ranges = [(Y_ARENA + j * y_step, Y_ARENA + (j + 1) * y_step) for j in range(ROW)]
    # print(f'RANGES: {x_ranges}, {y_ranges}')

    return x_ranges, y_ranges

def get_grid_index(X, Y, x_ranges, y_ranges):
    col = None
    row = None

    for i, (x_min, x_max) in enumerate(x_ranges):
        if x_min <= X < x_max:
            col = i
            break

    for j, (y_min, y_max) in enumerate(y_ranges):
        if y_min <= Y < y_max:
            row = j
            break

    return row, col

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

def calculate_grid_crossing(show_changes=False):
    x_ranges, y_ranges = calculate_grid_range()

    total_steps = {}
    for track_id in DATA['track'].unique(): # tranversing each track(instance)
        track_data = DATA[DATA['track'] == track_id]

        grid_positions = []
        for _, row in track_data.iterrows():
            X, Y = row[X_COL], row[Y_COL]

            if np.isnan(X) or np.isnan(Y):
                grid_positions.append(None)
                continue

            grid_pos = get_grid_index(X, Y, x_ranges, y_ranges)
            grid_positions.append(grid_pos)

        step_count = 0
        prev_grid = None
        prev_prev_grid = None

        for grid in grid_positions:
            if not is_valid_grid(grid):
                prev_prev_grid = prev_grid
                prev_grid = None
                continue

            # ignore immediate back-and-forth jitter
            if prev_prev_grid is not None and grid == prev_prev_grid:
                prev_prev_grid = prev_grid
                prev_grid = grid
                continue

            if prev_grid is not None and grid != prev_grid:
                step_count += 1
                if show_changes:
                    print(f'{track_id}: {prev_grid} → {grid} | Steps: {step_count}')

            prev_prev_grid = prev_grid
            prev_grid = grid

        total_steps[track_id] = step_count
        print(f'\n==================================Ended processing {track_id}==================================\n')
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


step_counts = calculate_grid_crossing(show_changes=True)
print("TOTAL STEPS PER MOUSE:")
for track, steps in step_counts.items():
    print(f'{track}: {steps}')

velocity_dict = calculate_instances_velocity()
print("\nAVERAGE VELOCITY PER MOUSE:")
for track, velocity in velocity_dict.items():
    print(f'{track}: {velocity:.2f} pixels/frame')