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

from mvn.models.model import LstmModel, TransformerModel, STGCNTransformerModel

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
            hidden_size=config.model.transformer_hidden_size, # 保持参数名一致性
            num_layers=config.model.n_layers,
            num_heads=config.model.n_heads,
            dropout_probability=config.model.dropout
        ).to(device)
    elif model_name == "stgcn_transformer":
        model = STGCNTransformerModel(
            seq_len=config.opt.seq_len,
            num_joints=config.opt.n_joints,
            stgcn_hidden_size=config.model.stgcn_hidden_size,
            transformer_hidden_size=config.model.transformer_hidden_size,
            num_layers=config.model.n_layers,
            num_heads=config.model.n_heads,
            dropout_probability=config.model.dropout
        ).to(device)
    elif model_name == "lstm":
        # Note: LstmModel的参数需要根据您的具体配置来确定
        model = LstmModel().to(device)
    else:
        raise NotImplementedError(f"模型 '{model_name}' 未实现。")

    # 如果指定了预训练权重，则加载
    if config.model.init_weights and config.model.checkpoint:
        print(f"从 '{config.model.checkpoint}' 加载预训练权重...")
        # 兼容性处理：只加载模型中存在的参数
        pretrained_dict = torch.load(config.model.checkpoint, map_location=device, weights_only=True)
        model_dict = model.state_dict()
        # 1. 将预训练权重中与当前模型匹配的键筛选出来
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. 更新当前模型的 state_dict
        model_dict.update(pretrained_dict)
        # 3. 加载我们真正需要的 state_dict
        model.load_state_dict(model_dict)
        print(f"成功加载 {len(pretrained_dict)} 个匹配的参数层。")

    return model
