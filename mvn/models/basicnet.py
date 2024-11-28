import torch
from torch import nn

class LstmModel(nn.Module):
    """
        使用LSTM进行原本时间序列与未来的预测

        参数：
        - input_size: 输入的维度
        - hidden_size: 隐藏单元数
        - output_size: 输出维度
        - num_layers:  LSTM层的层数
    """
    def __init__(self, input_size=51, hidden_size=96, output_size=51, num_layers=2):
        super().__init__()
 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_o = nn.Linear(hidden_size, output_size)
        self.fc_n = nn.Linear(hidden_size, output_size)

    def forward(self, x): 
        x, (hn, _) = self.lstm(x)          # x is input, size: (batch, seq_len, input_size)
        b, s, h = x.shape
        # 处理每一个时间步输出
        x = x.reshape(b * s, h)
        x = self.fc_o(x)
        x = x.reshape(b, s, -1)
        # 预测下一个时间输出
        hn = hn[-1].reshape(b, h)
        x_n = self.fc_n(hn).unsqueeze(1)    # Shape: (batch, 1, output_size)
        # 拼接为一个输出
        x = torch.cat((x, x_n), dim=1)
        return x
    
