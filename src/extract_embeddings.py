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


def self_blend(img, landmark, label):
    """Same blending as ESBI_Dataset.self_blending, extracted standalone so
    this script doesn't need to instantiate a full Dataset object."""
    target_landmark = landmark[48:68] if label == 2 else landmark
    mask = np.zeros_like(img[:, :, 0])
    try:
        cv2.fillConvexPoly(mask, cv2.convexHull(target_landmark.astype(int)), 1.)
    except Exception:
        mask = np.ones_like(img[:, :, 0])
    source = cv2.GaussianBlur(img.copy(), (5, 5), 0)
    blended = (img * (1 - mask[:, :, None]) + source * mask[:, :, None]).astype(np.uint8)
    return blended


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
    print(f"[INFO] Found {len(video_folders)} videos in {args.cropped_dir}")

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
            for f_path in sampled_paths:
                img = np.array(Image.open(f_path))

                if label != 0:
                    frame_name = os.path.basename(f_path).replace('.jpg', '.npy')
                    lm_path = os.path.join(args.landmark_dir, vid, frame_name)
                    if not os.path.exists(lm_path):
                        img_to_embed = img  # fall back to unblended frame if no landmark
                    else:
                        raw = np.load(lm_path, allow_pickle=True)
                        landmark = raw.item() if raw.dtype == 'O' or raw.ndim == 0 else raw
                        if landmark.ndim == 3:
                            landmark = landmark[0]
                        landmark = np.array(landmark).reshape(-1, 2)
                        img_to_embed = self_blend(img, landmark, label)
                else:
                    img_to_embed = img

                img_float = img_to_embed.astype('float32') / 255.0
                emb = embed_frame(model, img_float, device, args.image_size)
                embeddings.append(emb)

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
    parser.add_argument('--overwrite', action='store_true', help="Recompute embeddings even if the output file already exists.")
    args = parser.parse_args()
    main(args)
