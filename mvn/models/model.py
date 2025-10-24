'''
Date: 2024-12-09 18:42:46
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-18 16:20:50
Description: 最终输出模型
'''
import torch.nn as nn
import torch
import torch.nn.functional as F

from .basicnet import PositionalEncoding

class LstmModel(nn.Module):
    """
    使用LSTM进行原本时间序列与未来的预测。

    Args:
        feature_dimension (int, optional): 输入的维度。默认为51。
        hidden_size (int, optional): 隐藏单元数。默认为96。
        num_layers (int, optional): LSTM层的层数。默认为2。
    """
    def __init__(self, feature_dimension=51, hidden_size=96, num_layers=2):
        super().__init__()
 
        self.lstm = nn.LSTM(feature_dimension, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, feature_dimension)

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 `(batch_size, seq_len, feature_dimension)`。

        Returns:
            torch.Tensor: 输出张量，形状为 `(batch_size, seq_len, feature_dimension)`。
        """
        x, _ = self.lstm(x)          # x is input, size: (batch, seq_len, feature_dimension)
        b, s, h = x.shape
        # 处理每一个时间步输出
        x = x.reshape(b * s, h)
        x = self.fc(x)
        x = x.reshape(b, s, -1)
        return x

class TransformerModel(nn.Module):
    """
    基于Transformer的模型，用于预测人体姿态。

    Args:
        seq_len (int, optional): 序列长度。默认为5。
        num_joints (int, optional): 关节数量。默认为17。
        hidden_size (int, optional): 隐藏层大小。默认为96。
        num_layers (int, optional): Transformer编码器层数。默认为2。
        num_heads (int, optional): 多头注意力机制的头数。默认为8。
        dropout_probability (float, optional): dropout概率。默认为0.1。
    """
    def __init__(self, seq_len = 5, num_joints=17, hidden_size=96, num_layers=2, num_heads=8, dropout_probability=0.1):
        super().__init__()

        self.num_joints = num_joints

        # 输入嵌入层
        self.root_embedding = nn.Linear(3, hidden_size)
        self.rotation_embedding = nn.Linear(4 * (num_joints - 1), hidden_size)

        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_size, dropout_probability)

        # Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead=num_heads, dim_feedforward=2048, dropout=dropout_probability, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 输出层
        self.root_output_layer = nn.Linear(hidden_size, 3 * seq_len)
        self.rotation_output_layer = nn.Linear(hidden_size, (num_joints - 1) * 4 * seq_len)

    def forward(self, root, rotations):
        """
        前向传播。

        Args:
            root (torch.Tensor): 根节点位置，形状为 `(batch_size, seq_len, 3)`。
            rotations (torch.Tensor): 关节旋转，形状为 `(batch_size, seq_len, num_joints - 1, 4)`。

        Returns:
            tuple:
                - torch.Tensor: 预测的根节点位置，形状为 `(batch_size, seq_len, 3)`。
                - torch.Tensor: 预测的关节旋转，形状为 `(batch_size, seq_len, num_joints - 1, 4)`。
        """
        batch_size, seq_len, _ = root.shape

        # 1. 输入嵌入
        root_embed = self.root_embedding(root)  # (batch_size, seq_len, hidden_size)
        rotations_embed = self.rotation_embedding(rotations.view(batch_size, seq_len, -1))  # (batch_size, seq_len, hidden_size)

        # 2. 位置编码
        root_embed = self.pos_encoder(root_embed)
        rotations_embed = self.pos_encoder(rotations_embed)

        # 3. 特征拼接
        x = root_embed + rotations_embed  # (batch_size, seq_len, hidden_size)

        # 4. Transformer 编码器
        x = self.transformer_encoder(x) # (batch_size, seq_len, hidden_size)
        x = x[:, -1, :]

        # 5. 输出关节旋转信息
        rotation_predictions = self.rotation_output_layer(x).view(batch_size, seq_len, self.num_joints - 1, 4)
        normalized_rotation_predictions = F.normalize(rotation_predictions, p=2, dim=-1)

        # 6.输出根节点坐标
        root_predictions = self.root_output_layer(x).view(batch_size, seq_len, 3)

        return root_predictions, normalized_rotation_predictions
