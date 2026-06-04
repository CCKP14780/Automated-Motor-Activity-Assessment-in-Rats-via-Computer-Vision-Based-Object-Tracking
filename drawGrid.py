from locomotorActivity import *
import cv2

'''VIDEO = r'sleap-tutorial-data\mice.mp4'
OUTPUT_PATH = r'sleap-tutorial-data\output_grid_overlay.mp4'
'''
VIDEO = r'Result\\mice_new.mp4'
OUTPUT_PATH = r'Result\\output_grid_overlay.mp4'

COL = (0,0,255) # grid color
THICKNESS = 3

def display_frame_with_grid_overlay(save_video = False, flip = False):
    cap = cv2.VideoCapture(VIDEO)
    
    if not cap.isOpened():
        print('Video Not Found')
        exit()
    
    if save_video:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        if flip:
            global COL
            COL = (255,0,0)
            global OUTPUT_PATH
            OUTPUT_PATH = r'sleap-tutorial-data\flip_output_grid_overlay.mp4'
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
        print(f"Processing video... Saving to: {OUTPUT_PATH}")

    # convert every parameter to int
    X_AR = int(X_ARENA)
    Y_AR = int(Y_ARENA)
    W_AR = int(W_ARENA)
    H_AR = int(H_ARENA)

 # Initialize frame counter before the loop
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        # CRITICAL FIX: Check if frame was read successfully BEFORE manipulating it
        if not ret:
            print('End of Video')
            break

        # Flipping image for observers
        if flip:
            frame = cv2.flip(frame, 1)
            frame = cv2.flip(frame, 0)
        
        # Draw the arena bounding rectangle
        cv2.rectangle(frame, (X_AR, Y_AR), (X_AR + W_AR, Y_AR + H_AR), COL, 2)

        # Draw rows
        for row in range(1, ROW):
            row_start = (X_AR, Y_AR + (row * H_AR) // ROW)
            row_end = (X_AR + W_AR, Y_AR + (row * H_AR) // ROW)
            cv2.line(frame, row_start, row_end, COL, THICKNESS, cv2.LINE_AA)

        # Draw columns
        for col in range(1, COLUMN):
            col_start = (X_AR + (col * W_AR) // COLUMN, Y_AR)
            col_end = (X_AR + (col * W_AR) // COLUMN, Y_AR + H_AR)
            cv2.line(frame, col_start, col_end, COL, THICKNESS, cv2.LINE_AA)

        # --- DYNAMIC TORSO POSITION TRACKING ---
        # Filter data for the current frame index
        frame_data = DATA[DATA['frame_idx'] == frame_idx]
        
        for _, row in frame_data.iterrows():
            track_id = row['track']
            x = row[X_COL]
            y = row[Y_COL]
            
            # Ensure coordinates are valid numbers
            if pd.notna(x) and pd.notna(y):
                # Map tracks to colors (matching your original placeholders)
                if track_id == 'track_0':
                    color = (255, 0, 0)  # Blue in BGR
                elif track_id == 'track_1':
                    color = (0, 255, 0)  # Green in BGR
                else:
                    color = (0, 0, 255)  # Red default for other tracks
                
                # If the frame is flipped, adjust the coordinates accordingly
                if flip:
                    height, width, _ = frame.shape
                    x = width - 1 - x
                    y = height - 1 - y

                # Draw the dot representing the torso position
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)

        # Increment frame index for the next iteration
        frame_idx += 1

        if save_video:
            out.write(frame) # Write the modified frame to the file
        
        cv2.imshow('Video Playback', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

display_frame_with_grid_overlay(save_video=False, flip=False)
# display_frame_with_grid_overlay(save_video=True, flip=True)
