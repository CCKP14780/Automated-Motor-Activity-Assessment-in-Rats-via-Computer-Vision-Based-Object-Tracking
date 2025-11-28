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
INPUT_CSV = r'predictions\predictions\baseLine_labels.v002.000_mice_new.analysis.csv'
BODY_PART = 'torso'
X_COL, Y_COL = f'{BODY_PART}.x', f'{BODY_PART}.y'

def get_tile_indices(xc, yc, X_ARENA, Y_ARENA, tile_width, tile_height, ROW, COLUMN):
    """
    Calculates the 0-based row and column index of the grid tile 
    corresponding to a given (x, y) coordinate.
    
    This implements the core logic for the 'Unique Tiles Visited' metric.
    """
    
    # 1. Shift Origin (x_rel and y_rel): Coordinates relative to the arena's start
    x_rel = xc - X_ARENA 
    y_rel = yc - Y_ARENA
    
    # 2. Calculate Index (0-based) by normalizing and taking the floor
    col_idx = np.floor(x_rel / tile_width).astype(int)
    row_idx = np.floor(y_rel / tile_height).astype(int)
    
    # 3. Clip the indices to ensure they are within the valid grid boundaries [0, N-1]
    col_idx = np.clip(col_idx, 0, COLUMN - 1)
    row_idx = np.clip(row_idx, 0, ROW - 1)
    
    return row_idx, col_idx

def calculate_distance(group, X_COL, Y_COL):
    """
    Calculates the Total Euclidean distance traveled by a single track/mouse 
    by summing the pixel distance between consecutive frames.
    
    This implements the core logic for the 'Locomotor Activity' metric.
    """
    
    # 1. Calculate the difference (delta) in coordinates between consecutive frames
    # .diff() calculates the difference (x_i - x_{i-1}). .fillna(0) handles the first row.
    dx = group[X_COL].diff().fillna(0)
    dy = group[Y_COL].diff().fillna(0)
    
    # 2. Euclidean distance for each step: sqrt(dx^2 + dy^2)
    step_distance = np.sqrt(dx**2 + dy**2)
    
    # 3. Sum all the step distances to get the total distance
    total_distance = step_distance.sum()
    return total_distance


df = pd.read_csv(INPUT_CSV)
df_clean = df.dropna(subset=[X_COL, Y_COL]).copy()

TILE_WIDTH = W_ARENA // COLUMN
TILE_HEIGHT = H_ARENA // ROW

# 1. Tile Coverage: Calculate Tile Indices for every frame
df_clean['tile_row'], df_clean['tile_col'] = zip(*df_clean.apply(
        lambda r: get_tile_indices(r[X_COL], r[Y_COL], X_ARENA, Y_ARENA, TILE_WIDTH, TILE_HEIGHT, ROW, COLUMN), 
        axis=1
    ))

# Create a unique tile identifier (e.g., 'R0C5')
df_clean['tile_id'] = 'R' + df_clean['tile_row'].astype(str) + 'C' + df_clean['tile_col'].astype(str)

# Calculate Unique Tiles Visited per Track
unique_tiles_summary = df_clean.groupby('track')['tile_id'].nunique().reset_index(name='Unique Tiles Visited')

# 2. Locomotor Activity: Calculate Total Distance Traveled per Track
distance_summary = df_clean.groupby('track').apply(
        lambda group: calculate_distance(group, X_COL, Y_COL)
    ).reset_index(name='Total Distance (pixels)')

# 3. Merge and Report
final_report = unique_tiles_summary.merge(distance_summary, on='track').rename(columns={'track': 'Track ID'})

# Calculate Spatial Coverage Percentage
max_tiles = ROW * COLUMN
final_report['Spatial Coverage (%)'] = (final_report['Unique Tiles Visited'] / max_tiles) * 100

print("-" * 50)
print(f"OPEN FIELD BEHAVIORAL ANALYSIS (Tracking Body Part: {BODY_PART.upper()})")
print(f"Grid: {ROW} Rows x {COLUMN} Columns ({max_tiles} Total Tiles)")
print(f"Arena Box: X={X_ARENA}, Y={Y_ARENA}, W={W_ARENA}, H={H_ARENA}")
print("-" * 50)
print(final_report.to_markdown(index=False, numalign='left', stralign='left', floatfmt=".2f"))
print("-" * 50)
    
# Save the results
output_filename = INPUT_CSV.replace('.csv', f'_{BODY_PART}_analysis_report.csv')
final_report.to_csv(output_filename, index=False, float_format='%.2f')
print(f"Results saved to: {output_filename}")