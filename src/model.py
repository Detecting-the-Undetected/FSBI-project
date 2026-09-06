import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from utils.sam import SAM
import torchvision.models as models
import pywt
from PIL import Image
import numpy as np

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        
        # Change num_classes from 2 to 3 to match the (Real, FaceSwap, Reenactment) architecture
        self.net = EfficientNet.from_pretrained("efficientnet-b5", advprop=True, num_classes=3)

        self.cel = nn.CrossEntropyLoss()
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, x, target):
        # SAM optimizer requires two forward/backward passes per step
        for i in range(2):
            pred = self(x)
            if i == 0:
                pred_first = pred

            loss = self.cel(pred, target)

            self.optimizer.zero_grad()
            loss.backward()

            if i == 0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)

        return pred_first