
import numpy as np
import cv2
import os
import sys

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from VideoSkeleton import VideoSkeleton
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)

# ----------------------------
# Utilities / Dataset
# ----------------------------

class SkeToImageTransform:
    """Convert a (reduced) Skeleton to a 'stickman' RGB image (uint8) of given size."""
    def __init__(self, image_size: int):
        self.imsize = image_size

    def __call__(self, ske: Skeleton):
        image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)  # draws reduced sticks + joints
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced: bool, source_transform=None, target_transform=None):
        """
        videoSke(VideoSkeleton): associates a skeleton with each stored frame (image on disk)
        ske_reduced(bool): reduced skeleton (13 joints x 2 = 26) or full (33x3 = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset:",
              "ske_reduced=", ske_reduced, "=>", (Skeleton.reduced_dim if ske_reduced else Skeleton.full_dim))

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        ske = self.videoSke.ske[idx]
        x = self.preprocessSkeleton(ske)

        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            y = self.target_transform(image)
        else:
            y = transforms.ToTensor()(image)
        return x, y

    def preprocessSkeleton(self, ske: Skeleton):
        if self.source_transform:
            x = self.source_transform(ske)
        else:
            arr = ske.__array__(reduced=self.ske_reduced).flatten()
            x = torch.from_numpy(arr).to(torch.float32).reshape(arr.shape[0], 1, 1)
        return x

    def tensor2image(self, normalized_image: torch.Tensor):
        """
        normalized_image: (3,H,W) in [-1,1] (because of Normalize(0.5,0.5))
        returns: float image in BGR, range [0,1], shape (H,W,3)
        """
        numpy_image = normalized_image.detach().cpu().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))  # (H,W,C) RGB
        denorm = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denorm = np.clip(denorm, 0.0, 1.0)
        bgr = cv2.cvtColor(np.array(denorm), cv2.COLOR_RGB2BGR)
        return bgr


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ----------------------------
# Networks
# ----------------------------

class GenNNSke26ToImage(nn.Module):
    """
    Input:  (B,26,1,1)
    Output: (B,3,64,64) in [-1,1] (tanh)
    """
    def __init__(self):
        super().__init__()
        self.input_dim = Skeleton.reduced_dim
        # DCGAN-like small generator
        self.model = nn.Sequential(
            # (B,26,1,1) -> (B,512,4,4)
            nn.ConvTranspose2d(self.input_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 4->8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8->16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16->32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32->64
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.apply(init_weights)
        print(self.model)

    def forward(self, z):
        return self.model(z)


class GenNNSkeImToImage(nn.Module):
    """
    Input:  (B,3,64,64) stick image in [-1,1]
    Output: (B,3,64,64) in [-1,1]
    A light U-Net-ish encoder/decoder (no explicit skip connections for simplicity).
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1), # 32 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),# 16 -> 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),# 8 -> 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Bottleneck
        self.mid = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), # 4 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),    # 32 -> 64
            nn.Tanh(),
        )
        self.apply(init_weights)
        print(self)

    def forward(self, z):
        x = self.enc(z)
        x = self.mid(x)
        x = self.dec(x)
        return x


# ----------------------------
# Training / Inference wrapper
# ----------------------------

class GenVanillaNN:
    """
    Generates an image of the target person from a new skeleton posture.
    optSkeOrImage:
        1 -> input is 26D reduced skeleton (B,26,1,1)
        2 -> input is stick image (B,3,64,64)
    """
    def __init__(self, videoSke, loadFromFile: bool = False, optSkeOrImage: int = 1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_size = 64

        if optSkeOrImage == 1:
            self.netG = GenNNSke26ToImage()
            src_transform = None  # default skeleton -> (26,1,1)
            self.filename = 'data/Dance/DanceGenVanillaFromSke26.pth'
        else:
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([
                SkeToImageTransform(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.filename = 'data/Dance/DanceGenVanillaFromSkeim.pth'

        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # output image in [-1,1] after normalization => generator last layer should be tanh
        self.dataset = VideoSkeletonDataset(
            videoSke,
            ske_reduced=True,
            target_transform=tgt_transform,
            source_transform=src_transform
        )
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True, num_workers=0)

        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory:", os.getcwd())
            self.netG = torch.load(self.filename, map_location=self.device)

        self.netG.to(self.device)

    def train(self, n_epochs: int = 20, lr: float = 2e-4):
        self.netG.train()
        # L1 tends to give sharper outputs than MSE in image-to-image
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(n_epochs):
            running = 0.0
            for i, (x, y) in enumerate(self.dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                y_hat = self.netG(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()

                running += float(loss.item())
                if (i + 1) % 50 == 0:
                    print(f"[{epoch+1}/{n_epochs}] step {i+1}/{len(self.dataloader)}  L1={running/50:.4f}")
                    running = 0.0

            # save checkpoint each epoch (keeps it simple for TP)
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            torch.save(self.netG, self.filename)
            print(f"Saved: {self.filename}")

        print("Finished Training")

    @torch.no_grad()
    def generate(self, ske: Skeleton):
        """Generate image (BGR float in [0,1]) from a Skeleton instance."""
        self.netG.eval()
        x = self.dataset.preprocessSkeleton(ske)   # (C,1,1) or (3,64,64)
        x = x.unsqueeze(0).to(self.device)         # add batch
        y_hat = self.netG(x)[0]                    # (3,64,64)
        return self.dataset.tensor2image(y_hat)


if __name__ == '__main__':
    force = False
    optSkeOrImage = 2  # 1: 26D skeleton, 2: stick image
    n_epoch = 50

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"

    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    gen = GenVanillaNN(targetVideoSke, loadFromFile=False, optSkeOrImage=optSkeOrImage)
    gen.train(n_epoch)

    # quick test on training poses
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        image = cv2.resize(image, (256, 256))
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
