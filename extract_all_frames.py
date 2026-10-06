import cv2
import os
import argparse
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def process_video(args):
    v_path, output_dir, stride = args
    video_name = v_path.stem
    save_path = Path(output_dir) / video_name

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
        if count % stride == 0:
            cv2.imwrite(str(save_path / f"{video_name}_{count:04d}.jpg"), frame)
        count += 1
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a folder of videos at a fixed stride.")
    parser.add_argument('--video-dir', required=True, help="Folder containing .mp4 videos (e.g. FF++'s c23/videos, or Celeb-DF's video folder).")
    parser.add_argument('--output-dir', required=True, help="Where to save extracted frames (one subfolder per video).")
    parser.add_argument('--stride', type=int, default=10, help="Save every Nth frame (default: 10).")
    parser.add_argument('--workers', type=int, default=6, help="Number of parallel processes (default: 6).")
    parser.add_argument('--glob', default='*.mp4', help="Video file pattern to match (default: *.mp4).")
    args = parser.parse_args()

    print(f"Checking for videos in: {args.video_dir}")
    video_paths = list(Path(args.video_dir).glob(args.glob))
    print(f"Found {len(video_paths)} videos.")

    print("Filtering already processed videos...")
    to_process = []
    for p in video_paths:
        s_path = Path(args.output_dir) / p.stem
        if not (s_path.exists() and len(os.listdir(s_path)) > 0):
            to_process.append(p)

    print(f"Videos remaining to extract: {len(to_process)}")

    if len(to_process) > 0:
        tasks = [(p, args.output_dir, args.stride) for p in to_process]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(tqdm(executor.map(process_video, tasks), total=len(tasks)))
        print("\nExtraction Complete!")
    else:
        print("\nAll videos are already extracted.")
