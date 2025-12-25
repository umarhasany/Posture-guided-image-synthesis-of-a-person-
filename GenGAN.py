
import os
import sys

import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms

from VideoSkeleton import VideoSkeleton
from Skeleton import Skeleton
from GenVanillaNN import VideoSkeletonDataset, SkeToImageTransform, GenNNSkeImToImage, init_weights


class Discriminator(nn.Module):
    """
    Conditional discriminator: receives concatenation of (condition, image) => 6x64x64
    Outputs a patch score map (PatchGAN-ish).
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # in: (B,6,64,64)
            nn.Conv2d(6, 64, 4, 2, 1),            # 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False), # 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0),           # 1
            # no sigmoid: we will use BCEWithLogitsLoss
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class GenGAN:
    """GAN-like image generator (pix2pix-style): condition = stick image, output = target person image."""
    def __init__(self, videoSke, loadFromFile=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.netG = GenNNSkeImToImage().to(self.device)
        self.netD = Discriminator().to(self.device)

        self.filenameG = 'data/Dance/DanceGenGAN_G.pth'
        self.filenameD = 'data/Dance/DanceGenGAN_D.pth'

        image_size = 64
        src_transform = transforms.Compose([
            SkeToImageTransform(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        tgt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, source_transform=src_transform, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True, num_workers=0)

        if loadFromFile and os.path.isfile(self.filenameG):
            print("GenGAN: Load G=", self.filenameG, "   CWD=", os.getcwd())
            self.netG = torch.load(self.filenameG, map_location=self.device).to(self.device)
        if loadFromFile and os.path.isfile(self.filenameD):
            print("GenGAN: Load D=", self.filenameD, "   CWD=", os.getcwd())
            self.netD = torch.load(self.filenameD, map_location=self.device).to(self.device)

    def train(self, n_epochs=20, lr=2e-4, lambda_l1=100.0):
        """
        Train a conditional GAN:
            D tries to classify (stick, real) as real and (stick, fake) as fake
            G tries to fool D + match target via L1
        """
        bce = nn.BCEWithLogitsLoss()
        l1 = nn.L1Loss()
        optG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))
        optD = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))

        self.netG.train()
        self.netD.train()

        for epoch in range(n_epochs):
            d_running = 0.0
            g_running = 0.0
            for i, (cond, real) in enumerate(self.dataloader):
                cond = cond.to(self.device)   # (B,3,64,64)
                real = real.to(self.device)   # (B,3,64,64)

                # -------------------------
                # Train Discriminator
                # -------------------------
                with torch.no_grad():
                    fake = self.netG(cond)

                real_pair = torch.cat([cond, real], dim=1)  # (B,6,64,64)
                fake_pair = torch.cat([cond, fake], dim=1)

                pred_real = self.netD(real_pair)
                pred_fake = self.netD(fake_pair)

                # labels with same shape as preds (PatchGAN)
                real_lbl = torch.ones_like(pred_real)
                fake_lbl = torch.zeros_like(pred_fake)

                lossD = 0.5 * (bce(pred_real, real_lbl) + bce(pred_fake, fake_lbl))
                optD.zero_grad()
                lossD.backward()
                optD.step()

                # -------------------------
                # Train Generator
                # -------------------------
                fake = self.netG(cond)
                fake_pair = torch.cat([cond, fake], dim=1)
                pred_fake_for_g = self.netD(fake_pair)

                lossG_adv = bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
                lossG_l1 = l1(fake, real) * lambda_l1
                lossG = lossG_adv + lossG_l1

                optG.zero_grad()
                lossG.backward()
                optG.step()

                d_running += float(lossD.item())
                g_running += float(lossG.item())

                if (i + 1) % 50 == 0:
                    print(f"[{epoch+1}/{n_epochs}] step {i+1}/{len(self.dataloader)}  "
                          f"D={d_running/50:.4f}  G={g_running/50:.4f}")
                    d_running = 0.0
                    g_running = 0.0

            os.makedirs(os.path.dirname(self.filenameG), exist_ok=True)
            torch.save(self.netG, self.filenameG)
            torch.save(self.netD, self.filenameD)
            print(f"Saved: {self.filenameG} and {self.filenameD}")

        print("Finished Training")

    @torch.no_grad()
    def generate(self, ske: Skeleton):
        """Generate image (BGR float in [0,1]) from a Skeleton instance."""
        self.netG.eval()
        cond = self.dataset.preprocessSkeleton(ske)  # (3,64,64)
        cond = cond.unsqueeze(0).to(self.device)
        out = self.netG(cond)[0]
        return self.dataset.tensor2image(out)


if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"

    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    # Train (or set loadFromFile=True to test a saved model)
    gen = GenGAN(targetVideoSke, loadFromFile=False)
    gen.train(n_epochs=10)

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        image = cv2.resize(image, (256, 256))
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
