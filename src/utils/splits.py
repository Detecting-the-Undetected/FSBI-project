"""
Single source of truth for train/val/test video splits.
Used by BOTH Stage 1 (ESBI_Dataset) and Stage 2 (VideoEmbeddingDataset) so the
two stages can never disagree about which videos are test videos.

Place at: src/utils/splits.py
"""
import os
import random


def parse_test_list(path):
    """
    Parse Celeb-DF's List_of_testing_videos.txt -> (real_stems, fake_stems).
    Real vs fake is decided by the folder in each path (Celeb-synthesis = fake),
    NOT by the numeric label column, so we don't depend on its encoding.
    """
    real, fake = set(), set()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel = line.split()[-1]
            stem = os.path.splitext(os.path.basename(rel))[0]
            if 'synthesis' in rel.lower():
                fake.add(stem)
            else:
                real.add(stem)
    return real, fake


def split_videos(video_ids, test_list=None, seed=5, val_fraction=0.15):
    """
    Returns (train_ids, val_ids, test_ids) as sorted lists of video ids.

    test_list given (Celeb-DF): the official list defines the test set; every
        other video forms a pool that is shuffled with a fixed seed and split
        into train/val. (Shuffled, not sliced, so val isn't just "the end of the
        alphabet", which here would be a single video source.)
    test_list is None (FF++): the standard 720/140/140 alphabetical split, or the
        same 72/14/14 proportions when the dataset is smaller (debug subsets).
    """
    ids = sorted(video_ids)
    n_total = len(ids)

    if test_list:
        real_test, fake_test = parse_test_list(test_list)
        test_stems = real_test | fake_test
        test_ids = [v for v in ids if v in test_stems]
        pool = [v for v in ids if v not in test_stems]
        rng = random.Random(seed)
        rng.shuffle(pool)
        n_val = int(len(pool) * val_fraction)
        val_ids = sorted(pool[:n_val])
        train_ids = sorted(pool[n_val:])
        print(f"[SPLIT] official list: {len(train_ids)} train / {len(val_ids)} val / "
              f"{len(test_ids)} test videos (list has {len(real_test)} real + "
              f"{len(fake_test)} fake).")
        if len(test_ids) == 0:
            print("[SPLIT][WARNING] none of the listed test videos were found among the "
                  "available videos. If this isn't a tiny debug subset, the video names "
                  "probably don't match the list, and NOTHING is being held out.")
        return train_ids, val_ids, test_ids

    if n_total >= 1000:
        n_train, n_val = 720, 140
    else:
        n_train = int(n_total * 0.72)
        n_val = int(n_total * 0.14)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]
