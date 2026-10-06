import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

def get_crop_box(landmarks, margin=1.3):
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)
    w, h = x_max - x_min, y_max - y_min
    center_x, center_y = x_min + w/2, y_min + h/2

    size = max(w, h) * margin
    return int(center_x - size/2), int(center_y - size/2), int(size)

def main(args):
    frame_dir = Path(args.frame_dir)
    landmark_dir = Path(args.landmark_dir)
    output_dir = Path(args.output_dir)

    video_folders = [f for f in frame_dir.iterdir() if f.is_dir()]
    print(f"Cropping faces for {len(video_folders)} videos...")

    for v_folder in tqdm(video_folders):
        save_path = output_dir / v_folder.name
        save_path.mkdir(parents=True, exist_ok=True)

        frames = sorted(v_folder.glob('*.jpg'))

        for f_path in frames:
            lm_path = landmark_dir / v_folder.name / f"{f_path.stem}.npy"
            if not lm_path.exists(): continue

            img = cv2.imread(str(f_path))
            landmarks = np.load(str(lm_path))

            x, y, size = get_crop_box(landmarks, args.margin)

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x + size), min(img.shape[0], y + size)

            face = img[y1:y2, x1:x2]
            if face.size == 0: continue

            face = cv2.resize(face, (args.crop_size, args.crop_size))
            cv2.imwrite(str(save_path / f_path.name), face)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop aligned faces from frames using detected landmarks.")
    parser.add_argument('--frame-dir', required=True, help="Folder of extracted frames (one subfolder per video).")
    parser.add_argument('--landmark-dir', required=True, help="Folder of landmark .npy files (mirrors frame-dir structure).")
    parser.add_argument('--output-dir', required=True, help="Where to save cropped face images.")
    parser.add_argument('--crop-size', type=int, default=384, help="Output crop resolution (default: 384).")
    parser.add_argument('--margin', type=float, default=1.3, help="Crop box margin multiplier (default: 1.3).")
    args = parser.parse_args()
    main(args)
