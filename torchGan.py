# !pip install torchgan
# !pip install tensorboard

import os
os.environ["TENSORBOARD_LOGGING"] = '1'
# use 'tensorboard --logdir=log' to visualize loss.

from torch.optim import *
import torchvision.transforms as transforms
from torchgan.models import *
from torchgan.losses import *
from torchgan.trainer import Trainer
import torch.utils.data as Data
from torchvision.datasets import ImageFolder

batch_size = 256
imgSize = 64

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(imgSize),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageFolder("gan_2img/", transform=data_transform)

dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


trainer = Trainer({"generator": {"name": ACGANGenerator, "args": {"out_channels": 3, "step_channels": 16}, "optimizer": {"name": Adam, "args": {"lr": 0.008, "betas": (0.5, 0.999)}}},
                   "discriminator": {"name": ACGANDiscriminator, "args": {"in_channels": 3, "step_channels": 16}, "optimizer": {"name": Adam, "args": {"lr": 0.001, "betas": (0.5, 0.999)}}}},
                  [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()],
                  sample_size=32, epochs=200, ncritic=-2,
                  checkpoints="torchgan_output/model/gan",
                  recon="torchgan_output/images",
                  log_dir='torchgan_output/log')

trainer(dataloader)