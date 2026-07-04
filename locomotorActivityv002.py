import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
import re
from gridCoordinates import Grid
from locomotorCounter import LocomotorCounter

# =========================
# Load config
# =========================
with open('data.json', 'r') as f:
    config = json.load(f)

PATH = config.get('PATH')
VIDEO = f"{PATH}{config.get('VIDEO')}"
INPUT_CSV = f"{PATH}{config.get('INPUT_CSV')}"
BODY_SCORE = config.get('BODY_PARTS') 
BODY_PARTS = list(BODY_SCORE.keys())

ROW = config.get('GRID_ROW')
COLUMN = config.get('GRID_COLUMN')
OUTPUT = config.get('OUTPUT')
START_FRAME = config.get('START_FRAME', 0)
END_FRAME = config.get('END_FRAME', None)

THRESHOLD = config.get('THRESHOLD', 2.1)  
CONFIRM_FRAMES = config.get('CONFIRM_FRAMES', 3)

padding = 10

DATA = pd.read_csv(INPUT_CSV)

BODY_COORDS = {
    part: {'x': f'{part}.x', 'y': f'{part}.y'}
    for part in BODY_PARTS
}

# =========================
# Arena definition
# =========================
x_min_data = DATA[BODY_COORDS['torso']['x']].min()
x_max_data = DATA[BODY_COORDS['torso']['x']].max()
y_min_data = DATA[BODY_COORDS['torso']['y']].min()
y_max_data = DATA[BODY_COORDS['torso']['y']].max()

left_extension = 0.1 * (x_max_data - x_min_data)

X_AR = int((x_min_data - padding) - left_extension)
Y_AR = int(y_min_data - padding)
W_AR = int((x_max_data - x_min_data) + (2 * padding) + left_extension)
H_AR = int((y_max_data - y_min_data) + (2 * padding))

grid = Grid(X_AR, X_AR + W_AR, Y_AR, Y_AR + H_AR, ROW, COLUMN)

# =========================
# Utility
# =========================
def get_latest_version(name=OUTPUT):
    folder_path = Path('Result')
    highest_version = -1

    for file in folder_path.glob(f'{name}_v*.mp4'):
        if file.is_file():
            version_match = re.search(rf'{name}_v(\d+)', file.name)
            if version_match:
                ver_num = int(version_match.group(1))
                highest_version = max(highest_version, ver_num)

    new_version_number = highest_version + 1
    return f'{PATH}{name}_v{new_version_number:03d}.mp4'


def extract_body_grids(row):
    """
    Convert one CSV row (one rat in one frame) into:
    {
        "head": (row,col) or None,
        "torso": (row,col) or None,
        "tail_base": (row,col) or None
    }
    """
    body_grids = {}

    for part in BODY_PARTS:
        x_col = BODY_COORDS[part]['x']
        y_col = BODY_COORDS[part]['y']

        x = row[x_col]
        y = row[y_col]

        if pd.notna(x) and pd.notna(y):
            body_grids[part] = grid.calculate_grid_coordinates(x, y)
        else:
            body_grids[part] = None

    return body_grids


# =========================
# Main display + counting loop
# =========================
def display_frame_with_grid_overlay(save_video=False, start_frame=0, end_frame=None):
    cap = cv2.VideoCapture(VIDEO)
    out = None

    if not cap.isOpened():
        print('Video Not Found')
        return

    if end_frame is None:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if save_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    COLOR = (0, 0, 255)
    THICKNESS = 2

    # counters per rat
    counters = {
    'track_0': LocomotorCounter(
        track_id='track_0',
        body_score=BODY_SCORE,
        threshold=THRESHOLD,
        confirm_frames=CONFIRM_FRAMES
    ),
    'track_1': LocomotorCounter(
        track_id='track_1',
        body_score=BODY_SCORE,
        threshold=THRESHOLD,
        confirm_frames=CONFIRM_FRAMES
    )
    }

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_idx >= end_frame:
            print(f'Video ended at frame {end_frame}')
            break

        ret, frame = cap.read()
        if not ret:
            print('End of Video')
            break

        # draw arena + grid
        cv2.rectangle(frame, (X_AR, Y_AR), (X_AR + W_AR, Y_AR + H_AR), COLOR, 2)

        for row_i in range(1, ROW):
            row_start = (X_AR, Y_AR + (row_i * H_AR) // ROW)
            row_end = (X_AR + W_AR, Y_AR + (row_i * H_AR) // ROW)
            cv2.line(frame, row_start, row_end, COLOR, THICKNESS, cv2.LINE_AA)

        for col_i in range(1, COLUMN):
            col_start = (X_AR + (col_i * W_AR) // COLUMN, Y_AR)
            col_end = (X_AR + (col_i * W_AR) // COLUMN, Y_AR + H_AR)
            cv2.line(frame, col_start, col_end, COLOR, THICKNESS, cv2.LINE_AA)

        # display defaults
        display_score = {
            'track_0': f"Track 0: {counters['track_0'].step_count}",
            'track_1': f"Track 1: {counters['track_1'].step_count}",
        }

        # rows for this frame (one row per rat)
        frame_data = DATA[DATA['frame_idx'] == frame_idx]

        for _, row in frame_data.iterrows():
            track_id = row['track']
            if track_id not in counters:
                continue

            # build body-part grid summary for this rat in this frame
            body_grids = extract_body_grids(row)

            # update step count
            counters[track_id].update(frame_idx, body_grids)

            # draw body parts
            color = (255, 0, 0) if track_id == 'track_0' else (0, 255, 0)

            for part in BODY_PARTS:
                x_col = BODY_COORDS[part]['x']
                y_col = BODY_COORDS[part]['y']
                x = row[x_col]
                y = row[y_col]

                if pd.notna(x) and pd.notna(y):
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                    cv2.putText(
                        frame, part, (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1
                    )

            display_score[track_id] = f"{track_id}: {counters[track_id].step_count}"

        cv2.putText(frame, display_score['track_0'], (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, display_score['track_1'], (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if save_video and out is not None:
            out.write(frame)

        cv2.imshow('Video Playback', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            if save_video and out is not None:
                out.release()
                if Path(OUTPUT_PATH).exists():
                    Path(OUTPUT_PATH).unlink()
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    # write logs
    with open('testTrackLog.txt', 'w', encoding='utf-8') as f:
        for track_id, counter in counters.items():
            f.write(f'===== {track_id} =====\n')
            f.write(counter.get_log())
            f.write('\n\n')

    # final console summary
    print('\nFINAL STEP COUNTS')
    for track_id, counter in counters.items():
        print(f'{track_id}: {counter.step_count}')


def main():
    global OUTPUT_PATH
    OUTPUT_PATH = get_latest_version(OUTPUT)
    display_frame_with_grid_overlay(
        save_video=False,
        start_frame=START_FRAME,
        end_frame=END_FRAME
    )


main()