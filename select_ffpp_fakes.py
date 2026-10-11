"""
Pick the real FF++ fake videos worth evaluating on: only those whose TARGET video is in our
held-out FF++ test split (same split_videos() the training used), so the backbone never saw
that footage as a real training video.

Run from the PROJECT ROOT (needs src/utils/splits.py):
  python select_ffpp_fakes.py --downloads-dir <where faceforensics_download_v4.py wrote> \
      --real-cropped-dir <.../fsbi-ff-data/cropped_faces> --output-dir <ffpp_fakes_videos>

--downloads-dir is the folder you gave to faceforensics_download_v4.py; it contains
manipulated_sequences/<Method>/c23/videos/*.mp4. Output: <output-dir>/<Method>/<A_B>.mp4
"""
import os, sys, shutil, argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from utils.splits import split_videos
from eval_utils import select_unseen_fakes, FFPP_METHOD_CLASS

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--downloads-dir', required=True)
    ap.add_argument('--real-cropped-dir', required=True, help="FF++ cropped_faces (one folder per REAL video)")
    ap.add_argument('--output-dir', required=True)
    a = ap.parse_args()

    _, _, test_ids = split_videos(sorted(os.listdir(a.real_cropped_dir)))
    print(f"[INFO] {len(test_ids)} held-out FF++ test videos (e.g. {test_ids[:3]} ... {test_ids[-3:]})")
    for method in FFPP_METHOD_CLASS:
        src = Path(a.downloads_dir) / 'manipulated_sequences' / method / 'c23' / 'videos'
        if not src.is_dir():
            print(f"[SKIP] {method}: {src} not found"); continue
        stems = [p.stem for p in src.glob('*.mp4')]
        keep = select_unseen_fakes(stems, test_ids)
        out = Path(a.output_dir) / method
        out.mkdir(parents=True, exist_ok=True)
        for s in keep:
            shutil.copy2(src / f"{s}.mp4", out / f"{s}.mp4")
        print(f"[OK] {method}: downloaded {len(stems)}, kept {len(keep)} with an unseen target -> {out}")
