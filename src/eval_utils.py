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
