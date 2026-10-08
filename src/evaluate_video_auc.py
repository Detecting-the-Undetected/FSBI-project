"""
Video-level real-vs-fake evaluation on Celeb-DF's official test list.

For every video in List_of_testing_videos.txt:
  * sample 32 frames (evenly spaced, deterministic)
  * embed each with the Stage 1 FSBI backbone (NO synthetic blending: these are real
    videos and real deepfakes, exactly what the paper's Celeb-DF number is measured on)
  * score it two ways:
        FSBI only        : mean over frames of the backbone's own P(fake)
        FSBI + Bi-LSTM   : Bi-LSTM over the 32-frame embedding sequence -> P(fake)
  * P(fake) = P(FaceSwap) + P(Reenactment)   (collapses the 3-class model to binary)
A video with no usable frames gets 0.5, as in the paper, so every test video counts.

Usage:
  python evaluate_video_auc.py --stage1-ckpt <stage1.tar> --bilstm-ckpt <bilstm.tar> \
     --test-list configs/List_of_testing_videos.txt \
     --real-cropped-dir <.../celebdf-data/cropped_faces> \
     --fake-cropped-dir <.../celebdf-faketest/cropped_faces> [--limit-per-class 10]

Place at: src/evaluate_video_auc.py   (needs src/eval_utils.py next to it)
"""
import os
import json
import argparse
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model import Detector
from bilstm_model import BiLSTMClassifier
from extract_embeddings import embed_frame
from eval_utils import sample_indices, p_fake, build_entries, compute_metrics
from utils.splits import parse_test_list


def load_state(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt['model'] if 'model' in ckpt else ckpt


@torch.no_grad()
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = Detector()
    backbone.load_state_dict(load_state(args.stage1_ckpt, device))
    backbone = backbone.to(device).eval()
    embed_dim = backbone.net._fc.in_features  # 2048 for EfficientNet-B5; read, don't assume

    bilstm = BiLSTMClassifier(embed_dim=embed_dim, hidden_dim=args.hidden_dim)
    bilstm.load_state_dict(load_state(args.bilstm_ckpt, device))
    bilstm = bilstm.to(device).eval()
    print(f"[INFO] backbone: {args.stage1_ckpt}\n[INFO] bilstm:   {args.bilstm_ckpt}  (embed_dim={embed_dim})")

    real_stems, fake_stems = parse_test_list(args.test_list)
    entries = build_entries(real_stems, fake_stems, args.real_cropped_dir,
                            args.fake_cropped_dir, args.limit_per_class)
    print(f"[INFO] evaluating {len(entries)} videos "
          f"({sum(1 for e in entries if e[1]==0)} real, {sum(1 for e in entries if e[1]==1)} fake)")

    stems, labels, s_fsbi, s_bi, fallback, reasons = [], [], [], [], [], []
    for stem, label, folder in tqdm(entries, desc="Evaluating"):
        frames = sorted(glob(os.path.join(folder, '*.jpg')))
        stems.append(stem); labels.append(label)

        if not frames:
            s_fsbi.append(0.5); s_bi.append(0.5); fallback.append(True)
            reasons.append('missing_folder' if not os.path.isdir(folder) else 'no_frames')
            continue

        idxs = sample_indices(len(frames), args.n_frames)
        cache, embs = {}, []
        for i in idxs:
            if i not in cache:  # short videos repeat frames; embed each only once
                img = np.array(Image.open(frames[i]).convert('RGB')).astype('float32') / 255.0
                cache[i] = embed_frame(backbone, img, device, args.image_size)
            embs.append(cache[i])
        seq = torch.from_numpy(np.stack(embs)).float().to(device)  # (n_frames, embed_dim)

        # FSBI only: the backbone's own per-frame head, averaged over the video
        frame_probs = torch.softmax(backbone.net._fc(seq), dim=1).cpu().numpy()
        s_fsbi.append(float(p_fake(frame_probs).mean()))

        # FSBI + Bi-LSTM: temporal model over the whole sequence
        probs = torch.softmax(bilstm(seq.unsqueeze(0)), dim=1).cpu().numpy()[0]
        s_bi.append(float(p_fake(probs)))
        fallback.append(False); reasons.append('')

    results = {
        "FSBI only (frame-level, averaged)": compute_metrics(labels, s_fsbi, fallback),
        "FSBI + Bi-LSTM": compute_metrics(labels, s_bi, fallback),
    }

    print("\n" + "=" * 64)
    for name, m in results.items():
        a_all, a_ex = m['auc_all_videos'], m['auc_excluding_fallback']
        print(f"{name}\n   AUC (all videos)           : {a_all if a_all is None else round(a_all, 4)}"
              f"\n   AUC (excluding 0.5 fallback): {a_ex if a_ex is None else round(a_ex, 4)}"
              f"\n   mean P(fake)  real={m['mean_score_real']}  fake={m['mean_score_fake']}")
    m0 = results["FSBI + Bi-LSTM"]
    print(f"\nvideos: {m0['n_videos']} ({m0['n_real']} real / {m0['n_fake']} fake) | "
          f"fell back to 0.5: {m0['n_fallback_0.5']}")
    print("=" * 64)

    if m0['n_videos'] and m0['n_fallback_0.5'] / m0['n_videos'] > 0.2:
        print("[WARNING] more than 20% of videos had no frames and were scored 0.5. This almost "
              "always means --real-cropped-dir / --fake-cropped-dir point at the wrong folder "
              "(e.g. one level too high or low). The AUC above is NOT trustworthy until fixed.")

    with open(args.output_json, 'w') as f:
        json.dump({
            "summary": results,
            "per_video": [dict(video=s, label=l, p_fake_fsbi=a, p_fake_bilstm=b,
                               fallback=fb, reason=r)
                          for s, l, a, b, fb, r in zip(stems, labels, s_fsbi, s_bi, fallback, reasons)],
        }, f, indent=2)
    print(f"[INFO] per-video scores saved to {args.output_json}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Video-level AUC on Celeb-DF's official test list.")
    p.add_argument('--stage1-ckpt', required=True)
    p.add_argument('--bilstm-ckpt', required=True)
    p.add_argument('--test-list', required=True)
    p.add_argument('--real-cropped-dir', required=True, help="cropped_faces folder holding the real test videos")
    p.add_argument('--fake-cropped-dir', required=True, help="cropped_faces folder holding the fake test videos")
    p.add_argument('--n-frames', type=int, default=32)
    p.add_argument('--image-size', type=int, default=380)
    p.add_argument('--hidden-dim', type=int, default=256, help="Must match train_bilstm.py's --hidden-dim")
    p.add_argument('--limit-per-class', type=int, default=None, help="Only the first N real + N fake videos (quick test)")
    p.add_argument('--output-json', default='eval_results.json')
    main(p.parse_args())
