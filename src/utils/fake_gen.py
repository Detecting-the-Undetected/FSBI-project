"""
Synthetic forgery generation (SBI-style, after Section 3.1 of the FSBI paper) shared by
Stage 1 (ESBI_Dataset) and Stage 2 (extract_embeddings.py). numpy + cv2 only.

Place at: src/utils/fake_gen.py

What it does (a reimplementation following the paper's description, NOT the original SBI
code, and with our own parameter ranges):
  1. colour (RGB / HSV / brightness / contrast) + down-scale + sharpen transforms are applied
     to EITHER the source or the target copy of the image (50/50)
  2. mask = convex hull of the landmarks; the SAME small affine (translation + scale) is
     applied to the source image and the mask -> landmark mismatch
  3. elastic deformation of the mask; two Gaussian smoothings with an erosion between them
  4. I_sbi = I_s * M + I_t * (1 - M)          (paper Eq. 1)
Fakes with a degenerate mask raise FakeGenError instead of silently returning an unchanged
image (the failure that hid the coordinate bug).
"""
import os
import numpy as np
import cv2

COORD_MARKER = 'COORDS_CROP.txt'      # written by crop_faces.py next to crop-space landmarks
MIN_MASK_FRAC = 0.005                  # a usable blend mask covers at least 0.5% of the image
FACESWAP, REENACT = 1, 2

_space_cache = {}
_warned_legacy = False


class FakeGenError(Exception):
    pass


# ----------------------------------------------------------------------------- landmarks
def get_crop_box(landmarks, margin=1.3):
    """Same box crop_faces.py cuts out of the full frame (before clamping to the frame)."""
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)
    w, h = x_max - x_min, y_max - y_min
    size = max(w, h) * margin
    return int(x_min + w / 2 - size / 2), int(y_min + h / 2 - size / 2), int(size)


def load_raw_landmarks(landmark_dir, vid, frame_stem):
    raw = np.load(os.path.join(landmark_dir, vid, frame_stem + '.npy'), allow_pickle=True)
    lm = raw.item() if raw.dtype == object or raw.ndim == 0 else raw
    lm = np.asarray(lm, dtype=np.float32)
    if lm.ndim == 3:
        lm = lm[0]
    return lm.reshape(-1, 2)


def is_crop_space(landmark_dir):
    if landmark_dir not in _space_cache:
        _space_cache[landmark_dir] = os.path.exists(os.path.join(landmark_dir, COORD_MARKER))
    return _space_cache[landmark_dir]


def legacy_to_crop(lm_abs, crop_size):
    """Map FULL-FRAME landmarks into the crop, assuming the crop box was not clamped by the
    frame border. Approximate for faces near the border -> regenerate with crop_faces.py."""
    x, y, size = get_crop_box(lm_abs)
    return (lm_abs - np.array([x, y], np.float32)) * (crop_size / float(size))


def load_landmarks(landmark_dir, vid, frame_stem, crop_size):
    """Landmarks in the pixel coordinates of the stored crop image."""
    global _warned_legacy
    lm = load_raw_landmarks(landmark_dir, vid, frame_stem)
    if is_crop_space(landmark_dir):
        return lm
    if not _warned_legacy:
        print(f"[fake_gen][WARNING] {landmark_dir} has no {COORD_MARKER}: treating landmarks as "
              f"FULL-FRAME coordinates and converting approximately. Regenerate them with "
              f"crop_faces.py --crop-landmark-dir for exact alignment.")
        _warned_legacy = True
    return legacy_to_crop(lm, crop_size)


# ----------------------------------------------------------------------------- parameters
def sample_params(rng):
    k = lambda: int(4 * rng.integers(1, 6) + 1)            # 5, 9, 13, 17, 21 (as in SBI)
    return dict(
        on_target=bool(rng.random() < 0.5),
        rgb_shift=rng.uniform(-15, 15, 3).astype(np.float32),
        hue=float(rng.uniform(-10, 10)), sat=float(rng.uniform(-0.25, 0.25)),
        val=float(rng.uniform(-0.25, 0.25)),
        brightness=float(rng.uniform(-0.2, 0.2)), contrast=float(rng.uniform(-0.2, 0.2)),
        downscale=float(rng.uniform(0.5, 1.0)) if rng.random() < 0.5 else 1.0,
        sharpen=bool(rng.random() < 0.3),
        tx=float(rng.uniform(-0.03, 0.03)), ty=float(rng.uniform(-0.015, 0.015)),
        scale=float(rng.uniform(0.95, 1 / 0.95)),
        k1=k(), k2=k(), sigma2=float(rng.uniform(5, 45)), mask_size=int(rng.integers(192, 257)),
        elastic_seed=int(rng.integers(0, 2 ** 31 - 1)),
        blend_ratio=float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0])),
    )


def jitter_params(base, rng):
    """Small frame-to-frame drift around one sequence's parameters (Stage 2 fake sequences):
    the forgery wobbles slightly over time instead of being re-rolled from scratch."""
    p = dict(base)
    p['tx'] = base['tx'] + float(rng.normal(0, 0.004))
    p['ty'] = base['ty'] + float(rng.normal(0, 0.002))
    p['scale'] = base['scale'] * (1 + float(rng.normal(0, 0.005)))
    p['blend_ratio'] = float(np.clip(base['blend_ratio'] + rng.normal(0, 0.05), 0.2, 1.0))
    p['rgb_shift'] = (base['rgb_shift'] + rng.normal(0, 2, 3)).astype(np.float32)
    p['elastic_seed'] = int(rng.integers(0, 2 ** 31 - 1))
    return p


# ----------------------------------------------------------------------------- transforms
def _color_resolution(img, p):
    out = np.clip(img.astype(np.float32) + p['rgb_shift'], 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + p['hue']) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + p['sat']), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + p['val']), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out = np.clip(out * (1 + p['contrast']) + 127.5 * p['brightness'], 0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    if p['downscale'] < 1.0:
        small = cv2.resize(out, (max(8, int(w * p['downscale'])), max(8, int(h * p['downscale']))))
        out = cv2.resize(small, (w, h))
    if p['sharpen']:
        out = np.clip(out + 0.5 * (out.astype(np.float32) - cv2.GaussianBlur(out, (0, 0), 2)),
                      0, 255).astype(np.uint8)
    return out


def _affine(img, mask, p):
    h, w = mask.shape
    s = p['scale']
    M = np.array([[s, 0, (1 - s) * w / 2 + p['tx'] * w],
                  [0, s, (1 - s) * h / 2 + p['ty'] * h]], np.float32)
    img2 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask2 = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img2, mask2


def _elastic(mask, seed, alpha=50.0, sigma=7.0):
    h, w = mask.shape
    r = np.random.default_rng(seed)
    dx = cv2.GaussianBlur(r.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(r.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(mask, xs + dx, ys + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _smooth_blend_mask(mask, p):
    """Like SBI's get_blend_mask: smooth at ~192-256 px (small kernels 5..21), erode, smooth
    again, resize back. Returns (mask_before_ratio, mask_after_ratio)."""
    H, W = mask.shape
    s = p['mask_size']
    m = cv2.resize(mask, (s, s))
    m = cv2.GaussianBlur(m, (p['k1'], p['k1']), 0)
    m = m / max(float(m.max()), 1e-6)
    eroded = np.where(m >= 0.95, 1.0, 0.0).astype(np.float32)   # erosion after the first filter
    if eroded.sum() > 0:
        m = eroded
    m = cv2.GaussianBlur(m, (p['k2'], p['k2']), p['sigma2'])
    m = m / max(float(m.max()), 1e-6)
    m = cv2.resize(m, (W, H))
    return m, m * p['blend_ratio']


# ----------------------------------------------------------------------------- public API
def make_fake(img, lm, fake_type, p):
    """img: HxWx3 uint8 RGB crop. lm: landmarks in THIS image's pixel coordinates.
    fake_type 1 = FaceSwap (whole-face hull), 2 = Reenactment (mouth hull).
    Returns the forged uint8 image, or raises FakeGenError."""
    if lm.shape[0] < 68:
        raise FakeGenError(f"expected 68 landmarks, got {lm.shape[0]}")
    h, w = img.shape[:2]
    pts = lm[48:68] if fake_type == REENACT else lm
    mask = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(pts).astype(np.int32)), 1.0)
    if mask.sum() / (h * w) < MIN_MASK_FRAC:
        raise FakeGenError("landmarks fall (almost) outside the image -> coordinate-space "
                           "mismatch between landmarks and crop")
    source, target = img.copy(), img.copy()
    if p['on_target']:
        target = _color_resolution(target, p)
    else:
        source = _color_resolution(source, p)
    source, mask = _affine(source, mask, p)
    mask = _elastic(mask, p['elastic_seed'])
    m_full, m = _smooth_blend_mask(mask, p)
    if m_full.sum() / (h * w) < MIN_MASK_FRAC / 2:
        raise FakeGenError("blend mask vanished after smoothing")
    m = m[..., None]
    return np.clip(source.astype(np.float32) * m + target.astype(np.float32) * (1 - m),
                   0, 255).astype(np.uint8)


def post_augment(img, rng, p_compress=0.5):
    """Applied to BOTH the real and the fake image during Stage 1 training so colour shifts and
    JPEG artefacts can't become a shortcut for telling them apart (paper: Preprocessing)."""
    out = np.clip((img.astype(np.float32) + rng.uniform(-8, 8, 3)) * (1 + rng.uniform(-0.1, 0.1))
                  + 127.5 * rng.uniform(-0.1, 0.1), 0, 255).astype(np.uint8)
    if rng.random() < p_compress:
        ok, enc = cv2.imencode('.jpg', cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 101))])
        out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out
