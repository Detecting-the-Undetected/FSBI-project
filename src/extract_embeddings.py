"""
Stage 2 prep: runs a trained Stage 1 FSBI backbone over each video's frames to
produce per-video embedding sequences for Bi-LSTM training.

Design note: Stage 1 data only contains REAL videos (fakes are generated
on-the-fly, per-frame, with a random blend type). Stage 2 needs a whole
32-frame sequence to carry ONE consistent label. So for each real video we
generate three separate embedding sequences:
  - label 0 (Real):        the original, unblended frames
  - label 1 (FaceSwap):    all 32 frames consistently full-face self-blended
  - label 2 (Reenactment): all 32 frames consistently mouth/nose self-blended
This triples the video count (one real video -> 3 labeled video-samples) but
gives the Bi-LSTM a coherent temporal signal to learn from, rather than a
sequence with a different random label on every frame.

Usage:
    python extract_embeddings.py --checkpoint path/to/best.tar \
        --cropped-dir /path/to/cropped_faces --landmark-dir /path/to/landmarks \
        --output-dir /path/to/embeddings --n-frames 32
"""
import os
import argparse
import zlib
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pywt

from model import Detector
from utils import fake_gen


def get_dwt_rgb(x, wavelet='sym2', mode='reflect'):
    """Same FSBI frequency-fusion used in ESBI_Dataset — must match exactly,
    since the backbone was trained on images produced this way."""
    h, w = x.shape[:2]
    fused_channels = []
    for c in range(3):
        LL, (LH, HL, HH) = pywt.dwt2(x[:, :, c], wavelet, mode=mode)
        LL_resized = cv2.resize(LL.astype('float32'), (w, h), interpolation=cv2.INTER_LINEAR)
        fused = (LL_resized + x[:, :, c]) / 2.0
        fused_channels.append(fused)
    return np.stack(fused_channels, axis=0).astype('float32')


@torch.no_grad()
def embed_frame(model, img_rgb_float, device, image_size):
    """img_rgb_float: HxWx3 float32 in [0,1]. Returns a 1D embedding vector."""
    img_resized = cv2.resize(img_rgb_float, (image_size, image_size))
    fsbi = get_dwt_rgb(img_resized)  # (3, H, W)
    tensor = torch.from_numpy(fsbi).unsqueeze(0).to(device).float()

    # Replicates EfficientNet's forward() up to (but not including) the final
    # classifier FC — this is the pooled feature vector the paper's CNN
    # produces per frame (2048-dim for B5, NOT 1280 — that's a B0-era number;
    # don't hardcode 1280 anywhere downstream).
    feats = model.net.extract_features(tensor)
    pooled = model.net._avg_pooling(feats)
    embedding = pooled.view(1, -1).squeeze(0)
    return embedding.cpu().numpy()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Detector()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    video_folders = sorted(os.listdir(args.cropped_dir))
    if args.max_videos is not None:
        video_folders = video_folders[:args.max_videos]
    print(f"[INFO] Processing {len(video_folders)} videos from {args.cropped_dir}"
          + (f" (capped at --max-videos {args.max_videos})" if args.max_videos else ""))

    random.seed(5)  # reproducible frame sampling / blend assignment

    for vid in tqdm(video_folders, desc="Extracting video embeddings"):
        frame_paths = sorted(glob(os.path.join(args.cropped_dir, vid, '*.jpg')))
        if len(frame_paths) == 0:
            continue

        # Sample exactly n_frames frames, matching the paper's inference
        # protocol ("a set consisting of 32 frames was extracted from each
        # video"). If a video has fewer frames than n_frames, repeat frames
        # to pad rather than skip the video entirely.
        if len(frame_paths) >= args.n_frames:
            indices = sorted(random.sample(range(len(frame_paths)), args.n_frames))
        else:
            indices = list(range(len(frame_paths)))
            while len(indices) < args.n_frames:
                indices.append(random.choice(range(len(frame_paths))))
            indices = sorted(indices)
        sampled_paths = [frame_paths[i] for i in indices]

        # Three labeled sequences from the same video, as described above.
        for label, suffix in [(0, 'real'), (1, 'faceswap'), (2, 'reenact')]:
            out_path = os.path.join(args.output_dir, f"{vid}__{suffix}.npy")
            if os.path.exists(out_path) and not args.overwrite:
                continue

            embeddings = []
            # One parameter set per sequence, small per-frame drift -> a coherent forgery over time
            seq_rng = np.random.default_rng(zlib.crc32(f"{vid}_{label}".encode()))
            base_params = fake_gen.sample_params(seq_rng) if label != 0 else None
            if base_params is not None:
                base_params['on_target'] = False   # real sequence is untouched, so never tint the whole frame
            for f_path in sampled_paths:
                img = np.array(Image.open(f_path).convert('RGB'))
                if label != 0:
                    stem = os.path.splitext(os.path.basename(f_path))[0]
                    landmark = fake_gen.load_landmarks(args.landmark_dir, vid, stem, img.shape[0])
                    img_to_embed = None
                    for attempt in range(6):   # a degenerate draw is re-rolled, never crashes the run
                        pr = fake_gen.jitter_params(base_params, seq_rng) if attempt == 0 \
                            else dict(fake_gen.sample_params(seq_rng), on_target=False)
                        try:
                            img_to_embed = fake_gen.make_fake(img, landmark, label, pr)
                            break
                        except fake_gen.FakeGenError:
                            continue
                    if img_to_embed is None:
                        raise RuntimeError(f"could not forge {vid}/{stem} (type {label}); check landmarks")
                else:
                    img_to_embed = img
                img_float = img_to_embed.astype('float32') / 255.0
                embeddings.append(embed_frame(model, img_float, device, args.image_size))

            seq = np.stack(embeddings, axis=0)  # (n_frames, embed_dim)
            np.save(out_path, {'embeddings': seq, 'label': label, 'video_id': vid})

    print(f"[INFO] Done. Embeddings saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract per-video frame embedding sequences for Stage 2 Bi-LSTM training.")
    parser.add_argument('--checkpoint', required=True, help="Path to a trained Stage 1 FSBI checkpoint (.tar).")
    parser.add_argument('--cropped-dir', required=True, help="Folder of cropped face frames (one subfolder per video).")
    parser.add_argument('--landmark-dir', required=True, help="Folder of landmark .npy files (mirrors cropped-dir structure).")
    parser.add_argument('--output-dir', required=True, help="Where to save per-video embedding .npy files.")
    parser.add_argument('--n-frames', type=int, default=32, help="Frames per video sequence (default: 32, matching the paper's inference protocol).")
    parser.add_argument('--image-size', type=int, default=380, help="Must match the image_size the Stage 1 model was trained with.")
    parser.add_argument('--max-videos', type=int, default=None, help="Only process the first N videos (useful for quick plumbing tests without burning GPU quota on the whole dataset).")
    parser.add_argument('--overwrite', action='store_true', help="Recompute embeddings even if the output file already exists.")
    args = parser.parse_args()
    main(args)
