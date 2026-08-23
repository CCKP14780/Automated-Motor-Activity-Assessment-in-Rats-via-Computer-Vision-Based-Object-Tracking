import sleap_io as sio

labels = sio.load_slp(
    r"D:\GitHub\Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking\sleap-labeling\testPred.v001.slp"
)

print("Videos:", len(labels.videos))
print("Labeled frames:", len(labels.labeled_frames))

for i, video in enumerate(labels.videos):
    print("\nVideo", i)
    print(video)
    print("Filename:", video.filename)