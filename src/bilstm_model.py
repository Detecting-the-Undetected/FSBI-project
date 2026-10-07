import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class VideoEmbeddingDataset(Dataset):
    """
    Loads per-video embedding sequences saved by extract_embeddings.py.
    Each .npy file holds a dict: {'embeddings': (n_frames, embed_dim), 'label': int, 'video_id': str}.

    Split is percentage-based by unique video_id (not by file, since each
    video produced 3 files/labels) so the same source video's real/fake
    variants never leak across train/val/test — that would be a form of
    data leakage (the model could learn to recognize the *video*, not the
    manipulation artifact).
    """
    def __init__(self, embedding_dir, phase='train', train_ratio=0.72, val_ratio=0.14):
        self.embedding_dir = embedding_dir
        all_files = sorted(glob.glob(os.path.join(embedding_dir, '*.npy')))

        # Group files by underlying video_id (strip the __real/__faceswap/__reenact suffix)
        video_ids = sorted(set(os.path.basename(f).split('__')[0] for f in all_files))
        n_total = len(video_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        if phase == 'train':
            selected_ids = set(video_ids[:n_train])
        elif phase == 'val':
            selected_ids = set(video_ids[n_train:n_train + n_val])
        else:
            selected_ids = set(video_ids[n_train + n_val:])

        self.files = [f for f in all_files if os.path.basename(f).split('__')[0] in selected_ids]
        print(f"VideoEmbeddingDataset({phase}): {len(self.files)} video-samples "
              f"from {len(selected_ids)} unique videos (out of {n_total} total).")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        emb = torch.from_numpy(data['embeddings']).float()  # (n_frames, embed_dim)
        label = torch.tensor(data['label']).long()
        return emb, label


class BiLSTMClassifier(nn.Module):
    """
    Reads a (n_frames, embed_dim) sequence of per-frame FSBI embeddings,
    forward and backward via a bidirectional LSTM, and classifies the whole
    video into Real / FaceSwap / Reenactment.
    """
    def __init__(self, embed_dim, hidden_dim=256, num_layers=1, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, n_frames, embed_dim)
        lstm_out, _ = self.lstm(x)              # (batch, n_frames, hidden_dim*2)
        pooled = lstm_out.mean(dim=1)            # mean-pool over the time dimension
        return self.classifier(pooled)
