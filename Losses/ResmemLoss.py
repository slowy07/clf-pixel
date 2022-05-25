import torch
import sys
import os
from torch import nn, optim
from Losses.LossInterface import LossInterface
import resmem
from resmem import ResMem, transformer
from torchvision import transforms
from util import wget_file, map_number

recenter = transforms.Compose(
    (
        transforms.Resize((255, 255)),
        transforms.CenterCrop(227),
    )
)

resmem_url = "https://github.com/pixray/resmem/releases/download/1.1.3_model/model.pt"


class ResmemLoss(LossInterface):
    def __init__(self, **kwargs):
        if not os.path.exists(resmem.path):
            wget_file(resmem_url, resmem.path)
