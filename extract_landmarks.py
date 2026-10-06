import os
import cv2
import numpy as np
import argparse
import face_alignment
from tqdm import tqdm
from pathlib import Path

def main(args):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=args.device, compile=False)

    video_folders = [f for f in Path(args.frame_dir).iterdir() if f.is_dir()]
    print(f"Found {len(video_folders)} video folders. Starting landmark extraction...")

    for v_folder in tqdm(video_folders):
        save_path = Path(args.output_dir) / v_folder.name
        save_path.mkdir(parents=True, exist_ok=True)

        frames = sorted(v_folder.glob('*.jpg'))

        for f_path in frames:
            landmark_file = save_path / f_path.stem

            if os.path.exists(f"{landmark_file}.npy"):
                continue

            image = cv2.imread(str(f_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preds = fa.get_landmarks(image)

            if preds is not None:
                np.save(str(landmark_file), preds[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 2D facial landmarks for every frame in a dataset.")
    parser.add_argument('--frame-dir', required=True, help="Folder of extracted frames (one subfolder per video).")
    parser.add_argument('--output-dir', required=True, help="Where to save landmark .npy files (mirrors frame-dir structure).")
    parser.add_argument('--device', default='cuda', help="Device for face_alignment: 'cuda', 'mps', or 'cpu' (default: cuda).")
    args = parser.parse_args()
    main(args)
