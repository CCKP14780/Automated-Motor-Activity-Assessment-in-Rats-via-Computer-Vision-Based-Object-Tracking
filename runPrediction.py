import subprocess
'''
This Python utility runs prediction on the testing data.
'''

# Store the command directly as a string
import subprocess

powershell_cmd = r"""
sleap predict `
  --gui `
  --data_path "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/datasets/test/M15TestManualGray.mov" `
  --video_index 0 `
  --model_paths "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/models/260826_180558.centroid.n=408" `
  --model_paths "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/models/260826_200900.centered_instance.n=408" `
  --ensure_grayscale `
  --max_instances 1 `
  --track_matching_method hungarian `
  --tracking_window_size 20 `
  --candidates_method local_queues `
  --max_tracks 1 `
  --features centroids `
  --scoring_method euclidean_dist `
  -o "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/predictions/M15.fullvideo.n408.centroid.manual_grayscale.v001.predictions.slp"
"""

run_result_cmd = r"""
  uv run sleap-label "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/predictions/M15.fullvideo.n408.centroid.manual_grayscale.v001.predictions.slp"
"""
# Execute via PowerShell
subprocess.run(["powershell", "-Command", powershell_cmd], check=True)
subprocess.run(["powershell", "-Command", run_result_cmd], check=True)