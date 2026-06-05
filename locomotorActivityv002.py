import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# 1. import DATA and source video from config.
VIDEO = r'Result\\mice_new.mp4'
INPUT_CSV = r'Result\new_video.v002.000_mice_new.analysis.csv'
BODY_PARTS = ['head', 'torso', 'tail_base']
coords = {'x', 'y'}
DATA = pd.read_csv(INPUT_CSV)
BODY_COORDS = {
    part: {'x': f'{part}.x', 'y': f'{part}.y'}
    for part in BODY_PARTS
}

ROW = 3
COLUMN = 5
padding = 10

# base calculation (assuming rat thigmotaxis behavior)
x_min_data = DATA[BODY_COORDS['torso']['x']].min()
x_max_data = DATA[BODY_COORDS['torso']['x']].max()
y_min_data = DATA[BODY_COORDS['torso']['y']].min()
y_max_data = DATA[BODY_COORDS['torso']['y']].max()


#  2. draw grid overlay on video frames.
def display_frame_with_grid_overlay():
    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        print('Video Not Found')
        exit()

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
    frame_idx = 0  # Add frame counter

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

        # 3. track body positions and define mouse regions for each frames.
        frame_data = DATA[DATA['frame_idx'] == frame_idx]
        for _, row in frame_data.iterrows():
            for part in BODY_PARTS:
                x_col = BODY_COORDS[part]['x']
                y_col = BODY_COORDS[part]['y']
                x = row[x_col]
                y = row[y_col]
                
                if pd.notna(x) and pd.notna(y):
                    track_id = row['track']
                    color = (255, 0, 0) if track_id == 'track_0' else (0, 255, 0) if track_id == 'track_1' else (0, 0, 255)
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        
        cv2.imshow('Video Playback', frame)
        frame_idx += 1  # Increment frame counter

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 4. return result for evaluation.


display_frame_with_grid_overlay()
