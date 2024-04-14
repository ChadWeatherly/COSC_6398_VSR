import numpy as np
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build model architecture
class hr_feature_extractor(nn.Module):
    def __init__(self):
        super(hr_feature_extractor, self).__init__()
        # Define two 3D convolutional layers and max pooling layers

        # Layer 1: 2D Convolution
        # Input: (1, 64, 64) -> Output: (8, 64, 64)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Layer 2: 2D Convolution
        # Input: (8, 64, 64) -> Output: (16, 33, 33)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.maxpool1(self.conv2(x)))
        # print(x.size())
        return x


class lr_feature_extractor(nn.Module):
    def __init__(self):
        super(lr_feature_extractor, self).__init__()
        # Define three 2D convolutional layers and max pooling layers

        # Layer 1: 3D Convolution
        # Input: (1, 5, 32, 32) -> Output: (8, 5, 32, 32)
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Layer 2: 3D Convolution
        # Input: (8, 5, 32, 32) -> Output: (32, 6, 14, 14)
        self.conv2 = nn.Conv3d(8, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # Pass the input through the convolutional layers with ReLU activations
        x = self.relu1(self.conv1(x))
        # print(x.size())
        x = self.relu2(self.pool1(self.conv2(x)))
        # print(x.size())

        return x


class positional_encoder(nn.Module):
    def __init__(self):
        super(positional_encoder, self).__init__()
        # first layer
        self.fc1 = nn.Linear(1000, 500)
        self.relu1 = nn.ReLU()
        # second layer
        self.fc2 = nn.Linear(500, 1000)
        self.relu2 = nn.ReLU()

    def forward(self, t, T):
        # positional number to be used for positional encoding
        device = 'cpu'
        t = t.to(device)
        T = T.to(device)
        dec = int((t / T) * 1000)  # should be 3-digit number
        dec = torch.tensor(dec).to(device)
        pos_vec = torch.zeros(1000).to(device)
        fill = torch.tensor(1.0).to(device)
        pos_vec[dec] = fill
        x = self.relu1(self.fc1(pos_vec)).to(device)
        x = self.relu2(self.fc2(x))
        return x


class diffusion_vsr(nn.Module):
    def __init__(self, s=torch.tensor(0.0008), train_T=torch.tensor(1000), infer_T=torch.tensor(50)):
        super(diffusion_vsr, self).__init__()
        # Parameters
        self.train_T = train_T.to(device)
        self.infer_T = infer_T.to(device)
        self.s = s.to(device)
        # feature extraction layers
        self.hr_net = hr_feature_extractor().to(device)
        self.lr_net = lr_feature_extractor().to(device)
        # positional encoding network
        self.positional_encoder = positional_encoder().to(device)
        # model layers
        self.fc1 = nn.Linear(67576, 16000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16000, 4096)
        self.relu2 = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        # Extra parameter for weighted concatenation
        self.w1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.randn(1), requires_grad=True)

    def cosine_beta_schedule(self, timesteps, s):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = (timesteps + 1).to(device)
        x = torch.linspace(0, timesteps, steps).to(device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.beta_sch = torch.clip(betas, 0.01, 0.99)
        self.alpha_bar_sch = torch.clip(alphas_cumprod, 0.01, 0.99)

    def add_noise(self, img, t):
        """
        Adds noise to an image tensor for a given timestep.

        Parameters:
        img (Tensor): The original image tensor.
        t (int or Tensor): The current timestep or tensor of timesteps.

        Returns:
        Tensor: The noised image tensor.
        """
        device = img.device
        scaled_img = img * torch.sqrt(self.alpha_bar_sch[t]).to(device)

        mu = 0.0
        sig2 = float(1.0-self.alpha_bar_sch[t])
        random_noise = torch.empty(1,64,64).normal_(mean=mu, std=sig2).to(device)
        noisy_img = scaled_img + (torch.sqrt(1 - self.alpha_bar_sch[t])).to(device) * random_noise
        return noisy_img

    def calc_train_steps(self):
        self.cosine_beta_schedule(self.train_T, self.s)
        self.T = self.train_T

    def calc_test_steps(self):
        self.cosine_beta_schedule(self.infer_T, self.s)
        self.T = self.infer_T

    def full_inference(self, lr_imgs):
        self.calc_test_steps()
        hr_noise = torch.randn((1, 64, 64), requires_grad=False).to(device)
        for t in range(self.T):
            hr_noise = self.forward(hr_noise, lr_imgs, t)
        return hr_noise

    def forward(self, hr_noise, lr_seq, t):
        # forward actual predicts the t-1 noise image
        norm_hr_noise = hr_noise / 255.0
        norm_lr_seq = lr_seq / 255.0
        hr_features = self.hr_net(norm_hr_noise).flatten()  # output size is 17424
        # print(hr_features.size())
        lr_features = self.lr_net(norm_lr_seq).flatten()  # output size is 49152
        # print(lr_features.size())
        pos_vec = self.positional_encoder(t, self.T)  # output size is 1000
        x = torch.concat((hr_features, lr_features, pos_vec), dim=0)  # output size is 67576
        # print(x.size())
        x = self.relu1(self.fc1(x))    # output 16000
        x = self.relu2(self.fc2(x))    # output 4096
        # print(x.size())
        x = x.reshape(1, 64, 64)

        # Perform convolution
        x = self.relu3(self.conv1(x))
        x = self.relu4(self.conv2(x))
        # print(x.size(), norm_hr_noise.size())
        x = (self.w1 * x) + (self.w2 * norm_hr_noise)
        x = torch.clamp(x, 0, 1)
        x = 255.0 * x
        return x
