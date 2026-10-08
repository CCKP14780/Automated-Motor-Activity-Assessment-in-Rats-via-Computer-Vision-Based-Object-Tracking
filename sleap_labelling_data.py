import pandas as pd
path = 'C:/Users/ICT68/Documents/GitHub/Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking/sleap-labeling/'
videos = ['wistar-mouse-rgb-n409-points.video0.csv', 'wistar-mouse-rgb-n409-points.video1.csv', 'wistar-mouse-rgb-n409-points.video2.csv']
for video in videos:
    df=pd.read_csv(path + video)
    print('Rows:', len(df))
    print('Frames:', df['frame_idx'].nunique())
    print('Videos:', df['video_path'].nunique())
    print(df.groupby('video_path')['frame_idx'].nunique())