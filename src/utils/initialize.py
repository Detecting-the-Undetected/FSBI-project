from glob import glob
import os
import json
import numpy as np

def init_ff(phase, level='frame', n_frames=8):
    # This path must be correct relative to D:\major project\FSBI-main
    dataset_path = 'data/FaceForensics++/original_sequences/youtube/c23/frames/'
    
    image_list = []
    label_list = []
    
    # 1. Load the JSON IDs
    json_path = f'data/FaceForensics++/{phase}.json'
    with open(json_path, 'r') as f:
        filelist = json.load(f)

    # 2. Iterate through your IDs and find the folders
    for vid_id in filelist:
        folder_path = os.path.join(dataset_path, vid_id)
        
        if os.path.exists(folder_path):
            images_temp = sorted(glob(os.path.join(folder_path, '*.png')))
            
            if len(images_temp) > 0:
                # Select frames evenly across the video
                if n_frames < len(images_temp):
                    indices = np.linspace(0, len(images_temp) - 1, n_frames)
                    images_temp = [images_temp[int(round(idx))] for idx in indices]
                
                image_list += images_temp
                label_list += [0] * len(images_temp)

    # This print will help us see exactly what is happening
    print(f"ESBI({phase}): Found {len(image_list)} images in {len(filelist)} videos.")
    return image_list, label_list

# Map init_cdf to init_ff so it doesn't crash if called
def init_cdf(phase, level='frame', n_frames=8):
    return init_ff(phase, level, n_frames)