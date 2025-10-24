# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/24 18:00
@Author  : gjhhh
@File    : st_gcn.py
@Desc    : 时空图卷积网络 (ST-GCN) 的核心模块
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class Graph:
    """
    定义人体骨架的图结构。

    这个类负责管理人体关节的连接关系，并生成用于图卷积的邻接矩阵。
    我们使用的是 Human3.6M 数据集的17个关节点。
    """
    def __init__(self, layout='human36m', strategy='uniform'):
        self.num_node = 17
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = [
            (1, 0), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5), (7, 0),
            (8, 7), (9, 8), (10, 9), (11, 8), (12, 11), (13, 12),
            (14, 8), (15, 14), (16, 15)
        ]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

        self.A = self.get_adjacency_matrix(strategy)

    def get_adjacency_matrix(self, strategy: str) -> np.ndarray:
        """
        根据指定的策略计算邻接矩阵。

        Args:
            strategy (str): 'uniform', 'distance', 或 'spatial'。

        Returns:
            np.ndarray: 归一化后的邻接矩阵。
        """
        if strategy == 'uniform':
            # 简单策略：如果两个节点相连，则权重为1
            A = np.zeros((self.num_node, self.num_node))
            for i, j in self.self_link:
                A[i, j] = 1
            for i, j in self.neighbor:
                A[i, j] = 1
            # 归一化
            A = A / np.sum(A, axis=0, keepdims=True)
            return A
        else:
            raise ValueError("不支持的邻接矩阵策略: {}".format(strategy))

class GraphConvolution(nn.Module):
    """
    图卷积层。

    Args:
        in_channels (int): 输入特征的通道数。
        out_channels (int): 输出特征的通道数。
        kernel_size (int): 卷积核大小 (这里通常为1)。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=1,
            padding=0,
            bias=False
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量, 形状 (N, C_in, T, V)。
            A (torch.Tensor): 邻接矩阵, 形状 (K, V, V)。

        Returns:
            torch.Tensor: 输出张量, 形状 (N, C_out, T, V)。
        """
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        # 核心图卷积操作: A * x
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

class STGCN_Block(nn.Module):
    """
    时空图卷积块 (ST-GCN Block)。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        stride (int, optional): 时间维度上的步长。默认为1。
        dropout (float, optional): Dropout概率。默认为0。
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        # 空间图卷积
        self.gcn = GraphConvolution(in_channels, out_channels)
        # 时间卷积
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (9, 1), # 时间卷积核
                (stride, 1),
                ((9 - 1) // 2, 0) # padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

        # 残差连接
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量, 形状 (N, C, T, V)。
            A (torch.Tensor): 邻接矩阵。

        Returns:
            torch.Tensor: 输出张量。
        """
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)
