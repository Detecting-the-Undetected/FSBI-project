import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- CONFIG ---
FRAME_DIR = Path(r'D:\major project\FSBI-main\data\FaceForensics++\extracted_frames')
LANDMARK_DIR = Path(r'D:\major project\FSBI-main\data\FaceForensics++\landmarks')
OUTPUT_DIR = Path(r'D:\major project\FSBI-main\data\FaceForensics++\cropped_faces')
CROP_SIZE = 384
MARGIN = 1.3
# --------------

def get_crop_box(landmarks, margin=1.3):
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)
    w, h = x_max - x_min, y_max - y_min
    center_x, center_y = x_min + w/2, y_min + h/2
    
    size = max(w, h) * margin
    return int(center_x - size/2), int(center_y - size/2), int(size)

def main():
    video_folders = [f for f in FRAME_DIR.iterdir() if f.is_dir()]
    print(f"Cropping faces for {len(video_folders)} videos...")

    for v_folder in tqdm(video_folders):
        save_path = OUTPUT_DIR / v_folder.name
        save_path.mkdir(parents=True, exist_ok=True)
        
        frames = sorted(list(v_folder.glob('*.jpg')))
        
        for f_path in frames:
            lm_path = LANDMARK_DIR / v_folder.name / f"{f_path.stem}.npy"
            if not lm_path.exists(): continue
            
            # Load image and landmarks
            img = cv2.imread(str(f_path))
            landmarks = np.load(str(lm_path))
            
            # Calculate crop
            x, y, size = get_crop_box(landmarks, MARGIN)
            
            # Handle image boundaries
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x + size), min(img.shape[0], y + size)
            
            face = img[y1:y2, x1:x2]
            if face.size == 0: continue
            
            # Resize to final architecture target
            face = cv2.resize(face, (CROP_SIZE, CROP_SIZE))
            cv2.imwrite(str(save_path / f_path.name), face)

if __name__ == "__main__":
    main()