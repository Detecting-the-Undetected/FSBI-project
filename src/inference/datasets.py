import os

# --- Windows Path Config ---
BASE_PATH = r'D:\major project\FSBI-main\data\FaceForensics++'

def init_ff_3class(phase='train'):
    """
    Alignment for FSBI 3-class architecture:
    Uses only Real data from 'cropped_faces'.
    The SBI Generator will create Classes 1 and 2 during training.
    """
    image_list = []
    label_list = []

    # Path to your newly created cropped faces
    cropped_dir = os.path.join(BASE_PATH, 'cropped_faces')
    
    if os.path.exists(cropped_dir):
        # Load all 1,000 cropped video folders
        # We sort them to keep things consistent across training/val
        video_folders = sorted(os.listdir(cropped_dir))
        
        # Split logic (Standard FF++ split: 720 train, 140 val, 140 test)
        if phase == 'train':
            video_folders = video_folders[:720]
        elif phase == 'val':
            video_folders = video_folders[720:860]
        else: # test
            video_folders = video_folders[860:]

        for folder in video_folders:
            image_list.append(os.path.join(cropped_dir, folder))
            # Every folder here is 'Real' (Class 0).
            # Your training script will use these as 'Source' for fakes.
            label_list.append(0) 
    else:
        print(f"CRITICAL ERROR: Cropped directory not found at {cropped_dir}")

    print(f"Phase 1.5 Alignment: {phase.upper()} set ready with {len(image_list)} folders.")
    return image_list, label_list