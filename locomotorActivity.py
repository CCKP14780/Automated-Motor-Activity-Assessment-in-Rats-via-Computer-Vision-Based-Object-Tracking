import gridDetection as grid
import pandas as pd
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt

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

def calculate_grid_range():
    x_step = W_ARENA / COLUMN
    y_step = H_ARENA / ROW

    x_ranges = [(X_ARENA + i * x_step, X_ARENA + (i + 1) * x_step) for i in range(COLUMN)]
    y_ranges = [(Y_ARENA + j * y_step, Y_ARENA + (j + 1) * y_step) for j in range(ROW)]
    print(x_ranges, y_ranges)

    return x_ranges, y_ranges

def calculate_grid_crossing():
    #load data
    data = pd.read_csv(INPUT_CSV)
    location = data[['track', X_COL, Y_COL]].copy()

    # Getting total number of tracks and frames
    total_tracks = location['track'].nunique()
    total_frames = data['frame_idx'].max()
    print(f'Total Tracks: {total_tracks}, Total Frames: {total_frames}')

def calculate_instances_velocity():
    pass

calculate_grid_range()