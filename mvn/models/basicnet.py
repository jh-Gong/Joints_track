'''
Date: 2024-11-27 11:40:09
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-24 19:25:29
Description: example
'''
import torch
from torch import nn
import math
    
class PositionalEncoding(nn.Module):
    """
    位置编码函数。
    为输入的序列添加位置信息。
    """
    def __init__(self, model_dimension, dropout_probability=0.1, maximum_length=5000):
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout_probability)

        position = torch.arange(maximum_length).unsqueeze(1)
        divergence_term = torch.exp(torch.arange(0, model_dimension, 2) * (-math.log(10000.0) / model_dimension))
        positional_encoding = torch.zeros(maximum_length, model_dimension) 
        positional_encoding[:, 0::2] = torch.sin(position * divergence_term) 
        positional_encoding[:, 1::2] = torch.cos(position * divergence_term) 

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        """
        参数:
            x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        """
        x = x + self.positional_encoding[:x.size(1)].unsqueeze(0) 
        return self.dropout_layer(x)
    
class ConfidenceLayer(nn.Module):
    """
    置信度层。
    根据输入的置信度值，确定输出的置信度值。
    """
    def __init__(self, model_dimension):
        super().__init__()
        self.confidence_layer = nn.Linear(model_dimension, 1)

    def forward(self, x):
        """
        参数:
            x: 输入张量，形状为 [batch_size, seq_len, feature_dim]
        """
        confidence = self.confidence_layer(x)
        confidence = torch.sigmoid(confidence)
        return confidence