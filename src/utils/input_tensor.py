import torch
from torch import nn


class InputTensor(nn.Module):
    def __init__(self, input, gt, device):
        super(InputTensor, self).__init__()
        self.input = input.float().to(device)
        self.gt = gt.float().to(device)
        self.length = self.input.shape[0]

    def forward(self, xs):
        with torch.no_grad():
            xs = xs * torch.tensor([self.length], device=xs.device).float()
            indices = xs.long()
            indices.clamp(min=0, max=self.length - 1)
            return self.input[indices], self.gt[indices]
