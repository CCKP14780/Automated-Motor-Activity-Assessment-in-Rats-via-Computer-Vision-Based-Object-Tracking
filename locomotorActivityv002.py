import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import re
from gridCoordinates import Grid

# import DATA and source video from config.
with open('data.json', 'r') as f:
    config = json.load(f)

PATH = config.get('PATH')
VIDEO = f'{PATH}{config.get('VIDEO')}'
INPUT_CSV = f'{PATH}{config.get('INPUT_CSV')}'
BODY_PARTS = list(config.get('BODY_PARTS').keys())
ROW = config.get('GRID_ROW')
COLUMN = config.get('GRID_COLUMN')
padding = 10

coords = {'x', 'y'}
DATA = pd.read_csv(INPUT_CSV)

BODY_COORDS = {
    part: {'x': f'{part}.x', 'y': f'{part}.y'}
    for part in BODY_PARTS
}

# base calculation (assuming rat thigmotaxis behavior)
x_min_data = DATA[BODY_COORDS['torso']['x']].min()
x_max_data = DATA[BODY_COORDS['torso']['x']].max()
y_min_data = DATA[BODY_COORDS['torso']['y']].min()
y_max_data = DATA[BODY_COORDS['torso']['y']].max()

# init grid coordinates
grid = Grid(x_min_data - padding, x_max_data + padding, y_min_data - padding, y_max_data + padding, ROW, COLUMN)

# tracking latest video version
def get_latest_version():
    folder_path = Path('Result')
    highest_version = -1
    
    # Scan files to find the absolute highest version index present
    for file in folder_path.glob('output_grid_overlay_v*.mp4'):
        if file.is_file():
            version_match = re.search(r'output_grid_overlay_v(\d+)', file.name)
            if version_match:
                ver_num = int(version_match.group(1))
                if ver_num > highest_version:
                    highest_version = ver_num
                    
    new_version_number = highest_version + 1
    return f'{new_version_number:03d}'

version = get_latest_version()
OUTPUT_PATH = f'{PATH}output_grid_overlay_v{version}.mp4'

#  draw grid overlay on video frames.
def display_frame_with_grid_overlay(save_video=False):
    cap = cv2.VideoCapture(VIDEO)
    out = None

    if not cap.isOpened():
        print('Video Not Found')
        exit()

    if save_video:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    left_extension = 0.1 * (x_max_data - x_min_data)
    X_ARENA = (x_min_data - padding) - left_extension
    Y_ARENA = y_min_data - padding
    W_ARENA = (x_max_data - x_min_data) + (2 * padding) + left_extension
    H_ARENA = (y_max_data - y_min_data) + (2 * padding)

    X_AR = int(X_ARENA)
    Y_AR = int(Y_ARENA)
    W_AR = int(W_ARENA)
    H_AR = int(H_ARENA)

    COLOR = (0, 0, 255)
    THICKNESS = 2
    frame_idx = 0  # frame counter

    while True:
        ret, frame = cap.read()

        if not ret:
            print('End of Video')
            break

        cv2.rectangle(frame, (X_AR, Y_AR), (X_AR + W_AR, Y_AR + H_AR), COLOR, 2)
        
        for row in range(1, ROW):
            row_start = (X_AR, Y_AR + (row * H_AR) // ROW)
            row_end = (X_AR + W_AR, Y_AR + (row * H_AR) // ROW)
            cv2.line(frame, row_start, row_end, COLOR, THICKNESS, cv2.LINE_AA)
        
        for col in range(1, COLUMN):
            col_start = (X_AR + (col * W_AR) // COLUMN, Y_AR)
            col_end = (X_AR + (col * W_AR) // COLUMN, Y_AR + H_AR)
            cv2.line(frame, col_start, col_end, COLOR, THICKNESS, cv2.LINE_AA)

        # track body positions and define mouse regions for each frames.
        frame_data = DATA[DATA['frame_idx'] == frame_idx]
        for _, row in frame_data.iterrows():
            for part in BODY_PARTS:
                x_col = BODY_COORDS[part]['x']
                y_col = BODY_COORDS[part]['y']
                x = row[x_col]
                y = row[y_col]
                
                if pd.notna(x) and pd.notna(y):
                    track_id = row['track']
                    if track_id == 'track_0':
                        color = (255, 0, 0)      
                    elif track_id == 'track_1':
                        color = (0, 255, 0)      
                    else:
                        color = (0, 0, 255)      
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                    cv2.putText(frame, f'{part}', (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

                    print(track_id, part)
                    calculate_grid_coordinates = grid.calculate_grid_coordinates(x, y, part)
                    with open('grid_coordinates_log.txt', 'a') as f:
                        f.write(f'Frame {frame_idx}, {track_id}, {part}: Grid Position {calculate_grid_coordinates}\n')
        
        if save_video and out is not None:
            out.write(frame)

        cv2.imshow('Video Playback', frame)
        frame_idx += 1  # Increment frame counter

        if cv2.waitKey(25) & 0xFF == ord('q'):
            # delete the partially saved video if user quits early
            if save_video and out is not None:
                out.release()
                if Path(OUTPUT_PATH).exists():
                    Path(OUTPUT_PATH).unlink()
            break
    
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

# return result for evaluation.


display_frame_with_grid_overlay(save_video=False)
