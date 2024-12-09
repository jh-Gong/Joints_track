import torch.nn as nn
import math

from .basicnet import PositionalEncoding

class LstmModel(nn.Module):
    """
        使用LSTM进行原本时间序列与未来的预测

        参数：
        - feature_dimension: 输入的维度
        - hidden_size: 隐藏单元数
        - output_size: 输出维度
        - num_layers:  LSTM层的层数
    """
    def __init__(self, feature_dimension=51, hidden_size=96, num_layers=2):
        super().__init__()
 
        self.lstm = nn.LSTM(feature_dimension, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, feature_dimension)

    def forward(self, x): 
        x, _ = self.lstm(x)          # x is input, size: (batch, seq_len, feature_dimension)
        b, s, h = x.shape
        # 处理每一个时间步输出
        x = x.reshape(b * s, h)
        x = self.fc(x)
        x = x.reshape(b, s, -1)
        return x

class TransformerModel(nn.Module):
    def __init__(self, feature_dimension=51, hidden_size=96, num_layers=2, num_heads=8, dropout_probability=0.1):
        super().__init__()

        self.model_dimension = hidden_size
        self.linear_mapping = nn.Linear(feature_dimension, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout_probability)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout_probability, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_size, feature_dimension)

    def forward(self, x):
        """
        x: 输入序列，形状为 [batch_size, seq_len, feature_dimension]
        """
        x = self.linear_mapping(x) * math.sqrt(self.model_dimension)
        x = self.positional_encoding(x)

        memory = self.transformer_encoder(x)

        output = self.output_layer(memory)
        return output