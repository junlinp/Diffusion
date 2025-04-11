import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import tqdm
import cv2
import einops
from cifar_dataloader import CifarDataset

image_size = 32
dataset = CifarDataset(train=False)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
device = torch.device("mps") if (torch.backends.mps.is_available()) else device

print(f"Device: {device}")
from diffusion_model import VAE, DDPM


model = DDPM(img_width=image_size, img_height=image_size, img_channel=3)

model.to(device)
opt = optim.Adam(model.parameters(),lr=2 * 1e-4)

for epoch in range(1024):
    mean_loss = 0 
    model.train()
    for batch in tqdm.tqdm(dataloader):
        x, y = batch
        x = x.to(torch.float32).to(device)
        # print(f"x.shape {x.shape}")
        loss = model.compute_loss(x)
        mean_loss += loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch} loss : {mean_loss / len(dataset)}")

    sample_size = 16
    model.eval()
    inference_result = model.inference(sample_size = sample_size, device=device)
    # print(f"inference_result: {inference_result.shape}")
    inference_result = einops.rearrange(
        inference_result, "(h1 w1) c h w -> (h h1) (w w1) c", h1=4, w1=4
    )
    print(f"first element {inference_result[0, 0, 0]}")
    cv2.imwrite(f"outputs/{epoch}.png", (inference_result.cpu().detach().numpy() * 0.5 + 0.5) * 255)


# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()
