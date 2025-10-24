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
from .st_gcn import STGCN_Block, Graph

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

class STGCNTransformerModel(nn.Module):
    """
    一个混合模型，结合了ST-GCN和Transformer来预测人体姿态。

    Args:
        seq_len (int): 序列长度。
        num_joints (int): 关节数量。
        stgcn_hidden_size (int): ST-GCN模块的隐藏层大小。
        transformer_hidden_size (int): Transformer模块的隐藏层大小。
        num_layers (int): Transformer编码器层数。
        num_heads (int): 多头注意力机制的头数。
        dropout_probability (float): dropout概率。
    """
    def __init__(self, seq_len=15, num_joints=17, stgcn_hidden_size=64, transformer_hidden_size=128, num_layers=4, num_heads=8, dropout_probability=0.1):
        super().__init__()

        self.num_joints = num_joints

        # ST-GCN部分
        graph = Graph()
        self.A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.stgcn_block = STGCN_Block(in_channels=4, out_channels=stgcn_hidden_size)

        # 嵌入层
        self.root_embedding = nn.Linear(3, transformer_hidden_size)
        # GCN输出的特征维度是 stgcn_hidden_size * (num_joints - 1)
        self.rotation_embedding = nn.Linear(stgcn_hidden_size * (num_joints - 1), transformer_hidden_size)

        # 位置编码
        self.pos_encoder = PositionalEncoding(transformer_hidden_size, dropout_probability)

        # Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(transformer_hidden_size, nhead=num_heads, dim_feedforward=transformer_hidden_size * 4, dropout=dropout_probability, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 输出层
        self.root_output_layer = nn.Linear(transformer_hidden_size, 3 * seq_len)
        self.rotation_output_layer = nn.Linear(transformer_hidden_size, (num_joints - 1) * 4 * seq_len)

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
        batch_size, seq_len, _, _ = rotations.shape
        device = root.device

        # 1. ST-GCN空间特征提取
        # Reshape rotations for ST-GCN: (N, C, T, V)
        # N=batch_size, C=4 (quaternion), T=seq_len, V=num_joints-1
        rotations_gcn_input = rotations.permute(0, 3, 1, 2).contiguous()
        A = self.A.to(device)
        # GCN处理后, 输出形状: (N, stgcn_hidden_size, T, V)
        rotations_gcn_output = self.stgcn_block(rotations_gcn_input, A)

        # 2. 准备Transformer输入
        # Reshape GCN output for embedding layer
        rotations_gcn_output = rotations_gcn_output.permute(0, 2, 3, 1).contiguous() # (N, T, V, C_out)
        rotations_flat = rotations_gcn_output.view(batch_size, seq_len, -1) # (N, T, V*C_out)

        # 3. 输入嵌入
        root_embed = self.root_embedding(root)
        rotations_embed = self.rotation_embedding(rotations_flat)

        # 4. 位置编码
        root_embed = self.pos_encoder(root_embed)
        rotations_embed = self.pos_encoder(rotations_embed)

        # 5. 特征融合
        x = root_embed + rotations_embed

        # 6. Transformer 编码器
        x = self.transformer_encoder(x)
        x = x[:, -1, :] # 取最后一个时间步的输出作为序列的表示

        # 7. 输出层
        rotation_predictions = self.rotation_output_layer(x).view(batch_size, seq_len, self.num_joints - 1, 4)
        normalized_rotation_predictions = F.normalize(rotation_predictions, p=2, dim=-1)
        root_predictions = self.root_output_layer(x).view(batch_size, seq_len, 3)

        return root_predictions, normalized_rotation_predictions
