import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F # provides functions that don't need to be in a computational graph, i.e. aren't part of a NN, usually for single-use
import torchvision
from torch.utils.data import DataLoader, Dataset
from model import *
import matplotlib.pyplot as plt
import time as time
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
PYTORCH_NO_CUDA_MEMORY_CACHING=1
set('PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512')
set('PYTORCH_ENABLE_MPS_FALLBACK=1')

"""
if matplotlib doesn't run, go into the envs/pytorch_vsr environment in anaconda3
and delete all version of libiomp5md.dll , and it should work
"""

if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps") # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available
# device = 'cpu'
print(f"Using device: {device}")


def transform_image(image):
    # Performs downsampling and then blurring
    image = torchvision.transforms.functional.resize(image, (32, 32))
    gb = torchvision.transforms.GaussianBlur(kernel_size=(3, 3))
    image = gb(image)
    return image


def get_sample(data, idx):
    """
    Given a dataset (train/test) and a sample idx, a random sequence
    of 5 LR frames is computed
    """
    num_video_frames = data[idx].size()[0]
    rand_frame_idx = np.random.randint(2, num_video_frames - 2)
    # Get HR frame
    hr_frame = data[idx][rand_frame_idx].to(device)
    # Get LR frames
    lr_frames = [data[idx][k] for k in range(rand_frame_idx - 2, rand_frame_idx + 3)]
    lr_frames = [transform_image(frame) for frame in lr_frames]
    lr_frames = torch.stack(lr_frames, dim=0).to(device)  # permuted to have channel first
    return lr_frames, hr_frame

def ssim(img1, img2):
    k1 = torch.tensor(0.01, device=device)
    k2 = torch.tensor(0.03, device=device)
    mu1, sigma1 = img1.mean(), img1.std()
    mu2, sigma2 = img2.mean(), img2.std()
    cov = torch.sum( ( (img1 - mu1)*(img2 - mu2) ) / (64*64) )
    term1 = ( (2*mu1*mu2 + k1) / (mu1**2 + mu2**2 + k1) )
    term2 = ( (2*cov + k2) / (sigma1**2 + sigma2**2 + k2) )
    # print(term1, term2)
    return term1*term2

def psnr(img1, img2):
    mse = nn.MSELoss()(img1, img2)
    if mse==0:
        mse = torch.tensor(0.0000000000000001, device=device)
    return (10 * torch.log10(255.0**2 / mse))

# Load data
data = torchvision.datasets.MovingMNIST(root='./', split=None,
                                        split_ratio=10, download=True)
train_data = all_data[0:8000].to('cpu')      # 80% for train
test_data = all_data[8000:10001].to('cpu')    # 20% for test

# Get models
pq_recon_model = reconstructor(device=device).to(device).eval()
path = f'saved_models/recon_model_pq.pt'
old_model = torch.load(path, map_location=device)
pq_recon_model.load_state_dict(old_model[0])
train_pq_loss = old_model[1]
test_pq_loss = old_model[2]

mse_recon_model = reconstructor(device=device).to(device).eval()
path = f'saved_models/recon_model_mse.pt'
old_model = torch.load(path, map_location=device)
mse_recon_model.load_state_dict(old_model[0])
train_mse_loss = old_model[1]
test_mse_loss = old_model[2]

# Get avg. SSIM and PSNR
for metric in ['ssim', 'psnr']:
    for loss in ['mse', 'pq']:
        err = 0
        for idx in range(len(data)):
            video = data[idx]
            for frame in range(2, video.shape[0]-2):
                hr_img = video[frame].float().to(device)
                # print(hr_img.shape)
                lr_imgs = video[frame - 2:frame + 3]
                lr_imgs = [transform_image(img) for img in lr_imgs]
                lr_imgs = torch.stack(lr_imgs, dim=0).to(device)  # permuted to have channel first
                lr_imgs = lr_imgs.float()
                # print(lr_imgs.shape)

                # print(idx, frame, lr_imgs.shape)
                with torch.no_grad():
                    if loss == 'mse':
                        hr_pred = mse_recon_model(lr_imgs)
                    elif loss == 'pq':
                        hr_pred = pq_recon_model(lr_imgs)
                    # print(hr_img.shape, hr_pred.shape)
                    if metric == 'ssim':
                        err += ssim(hr_pred, hr_img)
                    elif metric == 'psnr':
                        err += psnr(hr_pred, hr_img)

                del hr_img, lr_imgs, hr_pred

            # if idx % 1000 == 0:
            #     print(metric, loss, idx)

        err = err / len(data)
        print(f'{loss} : {metric} : {err}')

# mse : ssim : 0.0023825119715183973
# pq : ssim : 0.00016504668747074902
# mse : psnr : 83.37740325927734
# pq : psnr : 83.24237823486328