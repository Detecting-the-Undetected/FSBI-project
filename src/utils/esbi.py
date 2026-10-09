import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import random
import cv2
from glob import glob
import pywt
from utils.funcs import crop_face
from utils.splits import split_videos
from utils import fake_gen

class ESBI_Dataset(Dataset):
    def __init__(self, phase='train', image_size=384, n_frames=8, wavelet="sym2", mode="reflect",
                 cropped_dir=None, landmark_dir=None, debug=False, test_list=None):
        self.phase = phase
        self.image_size = (image_size, image_size)
        self.w = wavelet
        self.m = mode

        self.path_lm = landmark_dir or '/kaggle/input/fsbi-ff-data/landmarks'
        cropped_dir = cropped_dir or '/kaggle/input/fsbi-ff-data/cropped_faces'

        video_folders = sorted(os.listdir(cropped_dir))
        n_total = len(video_folders)

        # Shared split logic (utils/splits.py) so Stage 1 and Stage 2 always agree.
        # FF++: standard 720/140/140. Celeb-DF: pass the official
        # List_of_testing_videos.txt via test_list and those videos are held out.
        train_ids, val_ids, test_ids = split_videos(video_folders, test_list)

        if debug:
            video_folders = video_folders[:1]
        elif phase == 'train':
            video_folders = train_ids
        elif phase == 'val':
            video_folders = val_ids
        else:
            video_folders = test_ids

        self.image_list = []
        for folder in video_folders:
            frames = sorted(glob(os.path.join(cropped_dir, folder, '*.jpg')))
            self.image_list.extend(frames)

        print(f'ESBI({phase}): Found {len(self.image_list)} images in {len(video_folders)} videos '
              f'(out of {n_total} total videos found in {cropped_dir}).')

    def get_dwt_rgb(self, x):
        """
        Implements the FSBI paper's Frequency Features Generator (Section 3.2):
        for each RGB channel, take the DWT approximate coefficient (LL), resize
        it back to the original resolution, and average it with the original
        channel. Stack the three fused channels depthwise.
        """
        h, w = x.shape[:2]
        fused_channels = []
        for c in range(3):
            LL, (LH, HL, HH) = pywt.dwt2(x[:, :, c], self.w, mode=self.m)
            LL_resized = cv2.resize(LL.astype('float32'), (w, h), interpolation=cv2.INTER_LINEAR)
            fused = (LL_resized + x[:, :, c]) / 2.0
            fused_channels.append(fused)
        img_fsbi = np.stack(fused_channels, axis=0)  # (3, H, W)
        return img_fsbi.astype('float32')

    def __getitem__(self, idx):
        attempts = 0
        while attempts < 10:
            try:
                filename = self.image_list[idx]
                img = np.array(Image.open(filename).convert('RGB'))
                vid = os.path.basename(os.path.dirname(filename))
                stem = os.path.splitext(os.path.basename(filename))[0]
                # landmarks in THIS crop's pixel coordinates (see utils/fake_gen.py)
                landmark = fake_gen.load_landmarks(self.path_lm, vid, stem, img.shape[0])

                rng = np.random.default_rng()
                fake_type = fake_gen.FACESWAP if rng.random() < 0.5 else fake_gen.REENACT
                img_f = fake_gen.make_fake(img, landmark, fake_type, fake_gen.sample_params(rng))
                img_r = img
                if self.phase == 'train':   # same augmentation on real and fake (paper)
                    img_f = fake_gen.post_augment(img_f, rng)
                    img_r = fake_gen.post_augment(img_r, rng)

                img_f = cv2.resize(img_f, self.image_size).astype('float32') / 255
                img_r = cv2.resize(img_r, self.image_size).astype('float32') / 255
                return self.get_dwt_rgb(img_f), self.get_dwt_rgb(img_r), fake_type
            except Exception as e:
                attempts += 1
                if attempts == 1 and not getattr(self, '_warned', False):
                    print(f'[ESBI] sample failed ({type(e).__name__}: {e}); retrying another frame')
                    self._warned = True
                idx = random.randint(0, len(self.image_list) - 1)
        raise RuntimeError('ESBI: 10 consecutive samples failed - check landmark/crop alignment')

    def collate_fn(self, batch):
        img_f, img_r, fake_types = zip(*batch)
        imgs = torch.cat([torch.tensor(np.array(img_r)), torch.tensor(np.array(img_f))], 0)
        labels = torch.tensor([0]*len(img_r) + list(fake_types))
        return {'img': imgs, 'label': labels}

    def worker_init_fn(self, worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def __len__(self): return len(self.image_list)
