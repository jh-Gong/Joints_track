import yaml
import json
from easydict import EasyDict as edict
from typing import Iterable, Tuple
import torch

def config_to_str(config: edict) -> str:
    """
    将配置对象转换为格式化的YAML字符串。

    Args:
        config (edict): 包含配置信息的EasyDict对象。

    Returns:
        str: 配置对象的YAML字符串表示形式。
    """
    # 在转换前先将 edict 转换为普通 dict，以获得更好的YAML格式
    return yaml.dump(json.loads(json.dumps(config)), indent=4)


class AverageMeter:
    """
    计算并存储一系列数值的平均值和当前值。
    """
    def __init__(self):
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0
        self.reset()

    def reset(self):
        """
        重置所有统计信息。
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        用新的值更新统计信息。

        Args:
            val (float): 当前值。
            n (int, optional): 值的数量。默认为1。
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_gradient_norm(named_parameters: Iterable[Tuple[str, torch.Tensor]]) -> float:
    """
    计算模型所有参数梯度的总范数 (L2范数)。

    Args:
        named_parameters (Iterable[Tuple[str, torch.Tensor]]): 包含命名参数的可迭代对象,
            通常是 `model.named_parameters()` 的输出。

    Returns:
        float: 梯度的总范数。
    """
    total_norm = 0.0
    for name, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5
    return total_norm
