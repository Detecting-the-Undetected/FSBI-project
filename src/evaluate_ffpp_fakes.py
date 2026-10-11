"""
Sanity check of the 3-class claim on REAL FF++ manipulations (not our synthetic fakes).

For every method folder under --fakes-root (Deepfakes, FaceSwap -> expected class 1;
Face2Face, NeuralTextures -> expected class 2) and the real FF++ test videos:
  * 32 evenly spaced frames per video, embedded with the Stage 1 backbone (no blending)
  * two scorers: FSBI only (frame-averaged 3-class probs) and FSBI + Bi-LSTM
  * report per method: AUC vs the real test videos, share flagged as fake, and the 3-class
    breakdown (does the head put swaps in class 1 and reenactments in class 2?)
Only fakes whose target video is in the held-out test split are used (select_unseen_fakes).

  python evaluate_ffpp_fakes.py --stage1-ckpt S1.tar --bilstm-ckpt B.tar \
     --real-cropped-dir <ffpp cropped_faces> --fakes-root <root with <Method>/<video>/*.jpg>
Place at: src/evaluate_ffpp_fakes.py
"""
import os, json, argparse
from glob import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model import Detector
from bilstm_model import BiLSTMClassifier
from extract_embeddings import embed_frame
from eval_utils import (sample_indices, p_fake, _safe_auc, summarize_probs,
                        select_unseen_fakes, FFPP_METHOD_CLASS)
from utils.splits import split_videos


def load_state(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt['model'] if 'model' in ckpt else ckpt


@torch.no_grad()
def score_video(folder, backbone, bilstm, device, n_frames, image_size):
    """-> (frame-averaged 3-class probs, Bi-LSTM 3-class probs) or None if no frames."""
    frames = sorted(glob(os.path.join(folder, '*.jpg')))
    if not frames:
        return None
    cache, embs = {}, []
    for i in sample_indices(len(frames), n_frames):
        if i not in cache:
            img = np.array(Image.open(frames[i]).convert('RGB')).astype('float32') / 255.0
            cache[i] = embed_frame(backbone, img, device, image_size)
        embs.append(cache[i])
    seq = torch.from_numpy(np.stack(embs)).float().to(device)
    p_frames = torch.softmax(backbone.net._fc(seq), dim=1).mean(0).cpu().numpy()
    p_bi = torch.softmax(bilstm(seq.unsqueeze(0)), dim=1).cpu().numpy()[0]
    return p_frames, p_bi


def main(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = Detector(); backbone.load_state_dict(load_state(a.stage1_ckpt, device))
    backbone = backbone.to(device).eval()
    bilstm = BiLSTMClassifier(embed_dim=backbone.net._fc.in_features, hidden_dim=a.hidden_dim)
    bilstm.load_state_dict(load_state(a.bilstm_ckpt, device)); bilstm = bilstm.to(device).eval()

    _, _, test_ids = split_videos(sorted(os.listdir(a.real_cropped_dir)))
    real_ids = test_ids[:a.limit] if a.limit else test_ids
    scorers = {"FSBI only": 0, "FSBI + Bi-LSTM": 1}

    def run(folders, desc):
        out = []
        for f in tqdm(folders, desc=desc):
            r = score_video(f, backbone, bilstm, device, a.n_frames, a.image_size)
            if r is not None:
                out.append(r)
        return out

    real = run([os.path.join(a.real_cropped_dir, s) for s in real_ids], "real test videos")
    real_probs = {k: np.array([r[i] for r in real]) for k, i in scorers.items()}
    print(f"[INFO] real test videos scored: {len(real)}/{len(real_ids)}")

    summary = {"real": {k: summarize_probs(real_probs[k], 0) for k in scorers}, "methods": {}}
    all_fake = {k: [] for k in scorers}
    for method, expected in FFPP_METHOD_CLASS.items():
        mdir = os.path.join(a.fakes_root, method)
        if not os.path.isdir(mdir):
            print(f"[SKIP] {method}: {mdir} not found"); continue
        stems = select_unseen_fakes([d for d in os.listdir(mdir) if os.path.isdir(os.path.join(mdir, d))], test_ids)
        if a.limit:
            stems = stems[:a.limit]
        res = run([os.path.join(mdir, s) for s in stems], method)
        if not res:
            print(f"[SKIP] {method}: no usable videos"); continue
        entry = {"n_videos": len(res), "expected_class": expected}
        for k, i in scorers.items():
            fp = np.array([r[i] for r in res]); all_fake[k].append(fp)
            labels = [0] * len(real_probs[k]) + [1] * len(fp)
            scores = list(p_fake(real_probs[k])) + list(p_fake(fp))
            entry[k] = {"auc_vs_real": _safe_auc(labels, scores), **summarize_probs(fp, expected)}
        summary["methods"][method] = entry

    summary["all_methods_pooled"] = {}
    for k in scorers:
        if all_fake[k]:
            fp = np.concatenate(all_fake[k])
            labels = [0] * len(real_probs[k]) + [1] * len(fp)
            summary["all_methods_pooled"][k] = _safe_auc(labels, list(p_fake(real_probs[k])) + list(p_fake(fp)))

    print("\n" + "=" * 78)
    print(f"{'method':<15}{'scorer':<16}{'n':>4}{'AUC':>8}{'flagged':>9}{'->swap':>8}{'->reenact':>11}{'type acc':>10}")
    for m, e in summary["methods"].items():
        for k in scorers:
            s = e[k]; ps = s['predicted_share']; ta = s['type_accuracy_among_flagged']
            print(f"{m:<15}{k:<16}{e['n_videos']:>4}{(s['auc_vs_real'] or float('nan')):>8.3f}"
                  f"{s['flagged_as_fake']:>9.2f}{ps['faceswap']:>8.2f}{ps['reenact']:>11.2f}"
                  f"{(ta if ta is not None else float('nan')):>10.2f}")
    print("pooled AUC (all methods vs real):", {k: (round(v, 4) if v is not None else None)
                                               for k, v in summary['all_methods_pooled'].items()})
    print("=" * 78)
    with open(a.output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] saved {a.output_json}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage1-ckpt', required=True); p.add_argument('--bilstm-ckpt', required=True)
    p.add_argument('--real-cropped-dir', required=True)
    p.add_argument('--fakes-root', required=True, help="folder with Deepfakes/ FaceSwap/ Face2Face/ NeuralTextures/ crop folders")
    p.add_argument('--n-frames', type=int, default=32); p.add_argument('--image-size', type=int, default=380)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--limit', type=int, default=None, help="quick test: first N videos per group")
    p.add_argument('--output-json', default='ffpp_realfakes_eval.json')
    main(p.parse_args())
