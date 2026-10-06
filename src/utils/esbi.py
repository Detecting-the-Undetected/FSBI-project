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

class ESBI_Dataset(Dataset):
    def __init__(self, phase='train', image_size=384, n_frames=8, wavelet="sym2", mode="reflect",
                 cropped_dir=None, landmark_dir=None, debug=False):
        self.phase = phase
        self.image_size = (image_size, image_size)
        self.w = wavelet
        self.m = mode

        self.path_lm = landmark_dir or '/kaggle/input/fsbi-ff-data/landmarks'
        cropped_dir = cropped_dir or '/kaggle/input/fsbi-ff-data/cropped_faces'

        video_folders = sorted(os.listdir(cropped_dir))
        n_total = len(video_folders)

        # FIX: splits are now percentage-based instead of hardcoded indices
        # (720/860) that silently broke / returned empty splits whenever the
        # dataset size changed (e.g. when you shrank it for faster iteration).
        # 80% train / 10% val / 10% test, by video count.
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)

        if debug:
            video_folders = video_folders[:1]
        elif phase == 'train':
            video_folders = video_folders[:n_train]
        elif phase == 'val':
            video_folders = video_folders[n_train:n_train + n_val]
        else:
            video_folders = video_folders[n_train + n_val:]

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

    def self_blending(self, img, landmark, label=None):
        # Class 2: Reenactment (Mouth/Nose only) | Class 1: FaceSwap (Full)
        target_landmark = landmark[48:68] if label == 2 else landmark
        mask = np.zeros_like(img[:,:,0])
        try:
            cv2.fillConvexPoly(mask, cv2.convexHull(target_landmark.astype(int)), 1.)
        except:
            mask = np.ones_like(img[:,:,0]) # Fallback
            
        source = img.copy() 
        source = cv2.GaussianBlur(source, (5,5), 0)
        img_blended = (img * (1 - mask[:,:,None]) + source * mask[:,:,None]).astype(np.uint8)
        return img, img_blended, mask

    def __getitem__(self, idx):
        attempts = 0
        while attempts < 10:
            try:
                filename = self.image_list[idx]
                img = np.array(Image.open(filename))
                
                vid = os.path.basename(os.path.dirname(filename))
                frame = os.path.basename(filename).replace('.jpg', '.npy')
                npy_path = os.path.join(self.path_lm, vid, frame)

                raw = np.load(npy_path, allow_pickle=True)
                landmark = raw.item() if raw.dtype == 'O' or raw.ndim == 0 else raw
                if landmark.ndim == 3: landmark = landmark[0]
                landmark = np.array(landmark).reshape(-1, 2)

                fake_type = 1 if random.random() < 0.5 else 2
                img_r, img_f, _ = self.self_blending(img, landmark, label=fake_type)
                
                img_f = cv2.resize(img_f, self.image_size).astype('float32') / 255
                img_r = cv2.resize(img_r, self.image_size).astype('float32') / 255
                
                return self.get_dwt_rgb(img_f), self.get_dwt_rgb(img_r), fake_type
            except Exception as e:
                attempts += 1
                idx = random.randint(0, len(self.image_list)-1)
        return torch.zeros(3, *self.image_size), torch.zeros(3, *self.image_size), 1

    def collate_fn(self, batch):
        img_f, img_r, fake_types = zip(*batch)
        imgs = torch.cat([torch.tensor(np.array(img_r)), torch.tensor(np.array(img_f))], 0)
        labels = torch.tensor([0]*len(img_r) + list(fake_types))
        return {'img': imgs, 'label': labels}

    def worker_init_fn(self, worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def __len__(self): return len(self.image_list)
