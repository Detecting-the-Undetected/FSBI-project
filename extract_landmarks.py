import os
import cv2
import numpy as np
import face_alignment
from tqdm import tqdm
from pathlib import Path

# --- CONFIG ---
FRAME_DIR = r'D:\major project\FSBI-main\data\FaceForensics++\extracted_frames'
# The SBI code usually looks for landmarks in a parallel folder
OUTPUT_DIR = r'D:\major project\FSBI-main\data\FaceForensics++\landmarks'
# --------------

def main():
    # Initialize the face aligner (Uses GPU by default if available)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda', compile=False)
    
    # Get all video folders
    video_folders = [f for f in Path(FRAME_DIR).iterdir() if f.is_dir()]
    print(f"Found {len(video_folders)} video folders. Starting landmark extraction...")

    for v_folder in tqdm(video_folders):
        save_path = Path(OUTPUT_DIR) / v_folder.name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Get all frames in this folder
        frames = sorted(list(v_folder.glob('*.jpg')))
        
        for f_path in frames:
            landmark_file = save_path / f_path.stem # will save as .npy
            
            # Skip if already exists
            if os.path.exists(f"{landmark_file}.npy"):
                continue
            
            # Read image and find landmarks
            image = cv2.imread(str(f_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preds = fa.get_landmarks(image)
            
            if preds is not None:
                # Save the first face detected as a numpy array
                np.save(str(landmark_file), preds[0])

if __name__ == "__main__":
    main()