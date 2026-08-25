import subprocess

# Store the command directly as a string
powershell_cmd = r"""
sleap predict `
  --data_path "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/datasets/test/M15.mov" `
  --model_paths "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/models/260825_115048.centroid.n=145/training_config.yaml" `
  --model_paths "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/models/260825_115048.centered_instance.n=145/training_config.yaml" `
  --ensure_grayscale `
  --peak_threshold 0.10 `
  --max_instances 1 `
  -o "C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/predictions/M15.mov_testPred.v002.slp"
"""

# Execute via PowerShell
subprocess.run(["powershell", "-Command", powershell_cmd], check=True)