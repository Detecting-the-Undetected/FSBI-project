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
    def __init__(self, phase='train', image_size=384, n_frames=8, wavelet="sym2", mode="reflect"):
        self.phase = phase
        self.image_size = (image_size, image_size)
        self.w = wavelet
        self.m = mode
        
        # Windows Absolute Paths
        self.path_lm = r'D:\major project\FSBI-main\data\FaceForensics++\landmarks'
        cropped_dir = r'D:\major project\FSBI-main\data\FaceForensics++\cropped_faces'
        
        video_folders = sorted(os.listdir(cropped_dir))
        if phase == 'train':
            video_folders = video_folders[:720]
        elif phase == 'val':
            video_folders = video_folders[720:860]
        else:
            video_folders = video_folders[860:]

        self.image_list = []
        for folder in video_folders:
            frames = sorted(glob(os.path.join(cropped_dir, folder, '*.jpg')))
            self.image_list.extend(frames)
        
        print(f'ESBI({phase}): Found {len(self.image_list)} images in {len(video_folders)} videos.')

    def get_dwt_rgb(self, x):
        # Wavelet Transform integration for your architecture
        coeffs = pywt.dwt2(x, self.w, mode=self.m)
        LL, (LH, HL, HH) = coeffs
        # Standardizing shapes for the model input
        return x.transpose(2, 0, 1) 

    def self_blending(self, img, landmark, label=None):
        # Class 2: Reenactment (Mouth/Nose only) | Class 1: FaceSwap (Full)
        target_landmark = landmark[48:68] if label == 2 else landmark
        mask = np.zeros_like(img[:,:,0])
        try:
            cv2.fillConvexPoly(mask, cv2.convexHull(target_landmark.astype(int)), 1.)
        except:
            mask = np.ones_like(img[:,:,0]) # Fallback
            
        source = img.copy() 
        # Apply a slight color shift to the source to create an artifact
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

                # Assign Class 1 or 2 for the fake image
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
        # Class 0: Real | Class 1: Swap | Class 2: Reenactment
        imgs = torch.cat([torch.tensor(np.array(img_r)), torch.tensor(np.array(img_f))], 0)
        labels = torch.tensor([0]*len(img_r) + list(fake_types))
        return {'img': imgs, 'label': labels}

    def worker_init_fn(self, worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def __len__(self): return len(self.image_list)