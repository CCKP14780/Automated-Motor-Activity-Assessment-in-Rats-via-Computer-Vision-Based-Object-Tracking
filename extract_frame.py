# Save this file to your project directory and run it
import cv2
import os

def extract_frames(video_path, output_folder, frame_skip=1):
    """
    Extracts frames from a video file and saves them as images.
    :param video_path: Path to the input video.
    :param output_folder: Folder where images will be saved.
    :param frame_skip: Extract every 'N' frames (1 means every frame).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_skip == 0:
            frame_name = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
            
        count += 1
        
    cap.release()
    print(f"Done! Successfully extracted {saved_count} frames to '{output_folder}'.")


extract_frames(r'sleap-labeling\datasets\train\M8.mov', r'sleap-labeling\datasets\stillFrames', frame_skip=5)
