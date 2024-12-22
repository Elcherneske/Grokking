import numpy
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        L2_fc = nn.MSELoss()
        return L2_fc(x, y)

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, score, target):
        CE_fc = nn.CrossEntropyLoss()
        loss = CE_fc(score, target)
        return loss

if __name__ == "__main__":
    import torch
    B = 64
    x = torch.randn(B, 10)
    y = torch.randn(B, 10)

    loss = L2Loss()

    print(x)
    print(y)
    print(loss(x, y))


