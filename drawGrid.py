from locomotorActivity import *
import cv2

VIDEO = r'sleap-tutorial-data\mice.mp4'
OUTPUT_PATH = r'sleap-tutorial-data\output_grid_overlay.mp4'
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

    while True:
        ret, frame = cap.read()

        # flipping image for observers
        if flip:
            frame = cv2.flip(frame,1)
            frame = cv2.flip(frame,0)


        if not ret:
            print('End of Video')
            break
        
        cv2.rectangle(frame, (X_AR, Y_AR),(X_AR+W_AR, Y_AR+H_AR), COL, 2)

        for row in range(1, ROW):
            row_start = (X_AR, Y_AR + (row * H_AR) // ROW)
            row_end = (X_AR + W_AR, Y_AR + (row * H_AR) // ROW)
            cv2.line(frame, row_start, row_end, COL, THICKNESS, cv2.LINE_AA)

        for col in range(1, COLUMN):
            col_start = (X_AR + (col * W_AR) // COLUMN, Y_AR)
            col_end = (X_AR + (col * W_AR) // COLUMN, Y_AR + H_AR)
            cv2.line(frame, col_start, col_end, COL, THICKNESS, cv2.LINE_AA)

        if save_video:
            out.write(frame) # Write the modified frame to the file
        
        cv2.imshow('Video Playback', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    if save_video:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


display_frame_with_grid_overlay(save_video=True, flip=False)
display_frame_with_grid_overlay(save_video=True, flip=True)