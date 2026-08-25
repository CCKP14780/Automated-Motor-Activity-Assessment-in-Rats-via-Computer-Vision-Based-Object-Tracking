#!/bin/bash
sleap train --config-name centroid.yaml --config-dir . 'trainer_config.ckpt_dir="models"' 'trainer_config.run_name="260825_115048.centroid.n=145"' 
sleap train --config-name centered_instance.yaml --config-dir . 'trainer_config.ckpt_dir="models"' 'trainer_config.run_name="260825_115048.centered_instance.n=145"' 
