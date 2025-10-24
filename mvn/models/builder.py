# -*- coding: utf-8 -*-

"""
@Time    : 2024/12/24 17:15
@Author  : gjhhh
@File    : builder.py
@Desc    : 模型构建器，用于根据配置构建模型
"""
from typing import Union
import torch
from torch import nn
from easydict import EasyDict as edict

from mvn.models.model import LstmModel, TransformerModel

def build_model(config: edict, device: torch.device) -> Union[nn.Module, None]:
    """
    根据配置构建并初始化模型。

    Args:
        config (edict): 包含模型配置的对象。
        device (torch.device): 模型将被加载到的设备 (例如, 'cuda' 或 'cpu')。

    Returns:
        Union[nn.Module, None]: 构建好的PyTorch模型。如果模型名称无法识别，则返回None。

    Raises:
        NotImplementedError: 如果配置文件中指定的模型名称未实现。
    """
    model_name = config.model.name

    if model_name == "transformer":
        model = TransformerModel(
            seq_len=config.opt.seq_len,
            num_joints=config.opt.n_joints,
            hidden_size=config.model.n_hidden_layer,
            num_layers=config.model.n_layers,
            num_heads=config.model.n_heads,
            dropout_probability=config.model.dropout
        ).to(device)
    elif model_name == "lstm":
        # Note: LstmModel的参数需要根据您的具体配置来确定
        # 这里使用默认参数作为示例
        model = LstmModel().to(device)
    else:
        raise NotImplementedError(f"模型 '{model_name}' 未实现。")

    # 如果指定了预训练权重，则加载
    if config.model.init_weights and config.model.checkpoint:
        print(f"从 '{config.model.checkpoint}' 加载预训练权重...")
        model.load_state_dict(torch.load(config.model.checkpoint, map_location=device, weights_only=True))

    return model
