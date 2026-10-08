"""
Copies ONLY the fake videos named in Celeb-DF's List_of_testing_videos.txt out of
the Celeb-synthesis folder, so you preprocess ~340 videos instead of thousands.
These are TEST-ONLY: keep them in their own folder tree, never in the training
cropped_faces/landmarks folders.

Usage:
  python select_test_fakes.py --list "D:\\...\\Celeb-DF\\List_of_testing_videos.txt" ^
      --synthesis-dir "C:\\Users\\PC\\Downloads\\Celeb-DF-v2\\Celeb-synthesis" ^
      --output-dir "D:\\major project\\FSBI-main\\data\\Celeb-DF\\fake_test\\videos"
"""
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--list', required=True, help="Path to List_of_testing_videos.txt")
parser.add_argument('--synthesis-dir', required=True, help="Folder holding the Celeb-synthesis videos")
parser.add_argument('--output-dir', required=True, help="Where to copy the selected fake test videos")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
wanted = []
with open(args.list) as f:
    for line in f:
        line = line.strip()
        if line and 'synthesis' in line.lower():
            wanted.append(os.path.basename(line.split()[-1]))

copied, missing = 0, []
for name in wanted:
    src = os.path.join(args.synthesis_dir, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(args.output_dir, name))
        copied += 1
    else:
        missing.append(name)

print(f"Fake videos in list: {len(wanted)} | copied: {copied} | missing: {len(missing)}")
if missing:
    print("First missing:", missing[:5])
