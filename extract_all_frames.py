import cv2
import os
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# --- PATHS ---
VIDEO_DIR = r'D:\major project\FSBI-main\data\FaceForensics++\original_sequences\youtube\c23\videos'
OUTPUT_DIR = r'D:\major project\FSBI-main\data\FaceForensics++\extracted_frames'
STRIDE = 10 
MAX_WORKERS = 6  # Boosted to 6 for your setup
# --------------

def process_video(v_path):
    video_name = v_path.stem
    save_path = Path(OUTPUT_DIR) / video_name
    
    # IMPROVED SKIP LOGIC:
    # Only skip if the folder exists AND it actually has files inside.
    if save_path.exists():
        if len(os.listdir(save_path)) > 0:
            return 
    
    save_path.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(v_path))
    count = 0
    while True:
        success, frame = cap.read()
        if not success: break
        if count % STRIDE == 0:
            # Saving as .png is slightly slower but higher quality for SBI, 
            # but .jpg is faster and uses less space. Sticking with .jpg for your drive.
            cv2.imwrite(str(save_path / f"{video_name}_{count:04d}.jpg"), frame)
        count += 1
    cap.release()

if __name__ == "__main__":
    print(f"Checking for videos in: {VIDEO_DIR}")
    video_paths = list(Path(VIDEO_DIR).glob('*.mp4'))
    print(f"Found {len(video_paths)} videos.")

    # Filter out what we already have before starting the pool (makes tqdm more accurate)
    print("Filtering already processed videos...")
    to_process = []
    for p in video_paths:
        s_path = Path(OUTPUT_DIR) / p.stem
        if not (s_path.exists() and len(os.listdir(s_path)) > 0):
            to_process.append(p)
    
    print(f"Videos remaining to extract: {len(to_process)}")

    if len(to_process) > 0:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            list(tqdm(executor.map(process_video, to_process), total=len(to_process)))
        print("\nExtraction Complete!")
    else:
        print("\nAll videos are already extracted.")