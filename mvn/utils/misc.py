import yaml
import json



def config_to_str(config):
    """
    将配置对象转换为字符串。

    Args:
        config (easydict.EasyDict): 包含配置信息的EasyDict对象。

    Returns:
        str: 配置对象的字符串表示形式。
    """
    return yaml.dump(yaml.safe_load(json.dumps(config)))


class AverageMeter(object):
    """
    计算并存储平均值和当前值。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        重置所有统计信息。
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        更新统计信息。

        Args:
            val (float): 当前值。
            n (int, optional): 值的数量。默认为1。
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_gradient_norm(named_parameters):
    """
    计算梯度的范数。

    Args:
        named_parameters (iterable): 包含命名参数的可迭代对象。

    Returns:
        float: 梯度的总范数。
    """
    total_norm = 0.0
    for name, p in named_parameters:
        # print(name)
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** (1. / 2)

    return total_norm
