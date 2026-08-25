import sleap_io as sio

# Load your standard labeling file
labels = sio.load_file("C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/wistar-mouse-grayscale-labels.v003.slp")

# Save it as a package file (embed=True copies the video frames inside)
labels.save("C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/wistar-mouse-grayscale-labels.v003.pkg.slp", embed=True)
