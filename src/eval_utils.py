"""
Pure-numpy helpers for video-level evaluation (no torch needed, so they are unit-testable).
Place at: src/eval_utils.py
"""
import os
import numpy as np
from sklearn.metrics import roc_auc_score


def sample_indices(n_available, n_frames):
    """Evenly spaced, deterministic frame indices (same videos -> same score every run).
    If a video has fewer frames than n_frames, indices repeat so we still return n_frames."""
    if n_available <= 0:
        return []
    return np.linspace(0, n_available - 1, n_frames).round().astype(int).tolist()


def p_fake(probs):
    """3-class probs [Real, FaceSwap, Reenactment] -> binary P(fake) = P(FaceSwap) + P(Reenactment).
    This collapses the 3-class model to the real-vs-fake task the paper reports on."""
    probs = np.asarray(probs)
    return probs[..., 1] + probs[..., 2]


def build_entries(real_stems, fake_stems, real_dir, fake_dir, limit_per_class=None):
    """-> list of (stem, label, folder). label: 0 = real, 1 = fake. Sorted, so a
    --limit run always takes the same videos."""
    real = sorted(real_stems)
    fake = sorted(fake_stems)
    if limit_per_class is not None:
        real, fake = real[:limit_per_class], fake[:limit_per_class]
    entries = [(s, 0, os.path.join(real_dir, s)) for s in real]
    entries += [(s, 1, os.path.join(fake_dir, s)) for s in fake]
    return entries


def _safe_auc(labels, scores):
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return None  # AUC is undefined with only one class present
    return float(roc_auc_score(labels, scores))


def compute_metrics(labels, scores, is_fallback):
    """
    labels: 0/1 per video. scores: P(fake) per video. is_fallback: True where the video
    had no usable frames and was given 0.5 (the paper's rule, so every test video counts).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    is_fallback = np.asarray(is_fallback, dtype=bool)
    keep = ~is_fallback
    return {
        "auc_all_videos": _safe_auc(labels, scores),
        "auc_excluding_fallback": _safe_auc(labels[keep], scores[keep]) if keep.any() else None,
        "n_videos": int(len(labels)),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "n_fallback_0.5": int(is_fallback.sum()),
        "mean_score_real": float(scores[labels == 0].mean()) if (labels == 0).any() else None,
        "mean_score_fake": float(scores[labels == 1].mean()) if (labels == 1).any() else None,
    }


CLASS_NAMES = ("real", "faceswap", "reenact")


def summarize_probs(video_probs, expected):
    """video_probs: (n_videos, 3) 3-class probabilities per video (mean over frames, or Bi-LSTM output).
    expected: the class index this group of videos should land in (0 real, 1 FaceSwap, 2 Reenactment)."""
    P = np.asarray(video_probs, dtype=float).reshape(-1, 3)
    n = len(P)
    if n == 0:
        return {"n_videos": 0}
    pred = P.argmax(1)
    fake_pred = pred != 0
    return {
        "n_videos": int(n),
        "expected_class": CLASS_NAMES[expected],
        "mean_probs": {c: float(P[:, i].mean()) for i, c in enumerate(CLASS_NAMES)},
        "predicted_share": {c: float((pred == i).mean()) for i, c in enumerate(CLASS_NAMES)},
        "flagged_as_fake": float((P[:, 1] + P[:, 2] > 0.5).mean()),
        "accuracy_expected_class": float((pred == expected).mean()),
        "type_accuracy_among_flagged": (float((pred[fake_pred] == expected).mean())
                                        if fake_pred.any() and expected != 0 else None),
    }


# ----------------------------------------------------------------------------- real FF++ fakes
# FF++ manipulation methods -> the class our 3-class head should assign them
#   identity swaps (a different face is pasted on)      -> 1 FaceSwap
#   reenactment (same identity, expression/mouth moved)  -> 2 Reenactment
FFPP_METHOD_CLASS = {"Deepfakes": 1, "FaceSwap": 1, "Face2Face": 2, "NeuralTextures": 2}


def fake_target_id(stem):
    """FF++ fake videos are named <target>_<source> (e.g. '860_123'): the target video is the
    background/identity the fake was built on. Returns '860'."""
    return stem.split('_')[0]


def select_unseen_fakes(video_stems, test_ids):
    """Keep only fakes whose TARGET video is in the held-out test split, so the backbone never saw
    that footage as a real training video."""
    test_ids = set(test_ids)
    return sorted(s for s in video_stems if fake_target_id(s) in test_ids)
