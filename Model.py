import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, mlp=[128, 512, 97]):
        """
        :param mlp: 一个列表，表示每一层的神经元数量，默认值为 [128, 512, 97]，
                    其中最后一层 97 是输出层的神经元数目。
        """
        super().__init__()
        # 构建 MLP 的层
        layers = []
        # 遍历输入的 mlp 列表并构建每一层
        for i in range(1, len(mlp)):
            layers.append(nn.Linear(mlp[i - 1], mlp[i]))  # 添加线性层
            if i != len(mlp) - 1:
                layers.append(nn.ReLU())  # 添加 ReLU 激活函数
        # 最后一层不加 ReLU 激活函数
        self.network = nn.Sequential(*layers)

    def forward(self, data):
        """
        :param x: 输入张量
        :return: 输出张量
        """
        feature = torch.cat([data['x'].unsqueeze(1), data['op'].unsqueeze(1), data['y'].unsqueeze(1), data['='].unsqueeze(1)], dim=1)
        feature = self.network(feature)
        feature = torch.mean(feature, dim = 1, keepdim=False)
        return feature




