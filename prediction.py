import sleap
import json

with open("path/to/model_config.json") as f:
    model_config = json.load(f)

predictor = sleap.load_model([model_config["centroid_model"], model_config["instance_model"]])
video = sleap.load_video("path/to/video.mp4")
predictions = predictor.predict(video)
predictions.export("local_predictions.slp")
