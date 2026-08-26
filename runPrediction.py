import subprocess
'''
This Python utility runs prediction on the testing data.
'''

# Store the command directly as a string
powershell_cmd = r"""
sleap predict `
  --data_path "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/datasets/train/F8.mov" `
  --model_paths "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/models/260825_231811.centroid.n=366/training_config.yaml" `
  --model_paths "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/models/260825_234050.centered_instance.n=366/training_config.yaml" `
  --ensure_grayscale `
  --peak_threshold 0.20 `
  --max_instances 1 `
  -o "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/predictions/F8.mov_testPred.peak_0_20.with_tracking.v003.slp"
"""

run_result_cmd = r"""
  uv run sleap-label "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/predictions/F8.mov_testPred.peak_0_20.with_tracking.v003.slp"
"""
# Execute via PowerShell
subprocess.run(["powershell", "-Command", powershell_cmd], check=True)
subprocess.run(["powershell", "-Command", run_result_cmd], check=True)