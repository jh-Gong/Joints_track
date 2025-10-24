import yaml
from easydict import EasyDict as edict
import argparse
import os

def load_config(path: str) -> edict:
    """
    加载YAML配置文件。

    Args:
        path (str): YAML配置文件的路径。

    Returns:
        edict: 包含配置信息的EasyDict对象。
    """
    with open(path) as f:
        config = edict(yaml.safe_load(f))

    return config

def parse_args(work_directory: str) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        work_directory (str): 工作目录的路径。

    Returns:
        argparse.Namespace: 包含解析后的命令行参数的对象。
    """
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加配置文件路径参数
    parser.add_argument("--config", type=str, default=None, help="Path, where config file is stored")
    # 添加评估模式参数
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    # 添加评估数据集参数
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")
    # 添加日志目录路径参数
    parser.add_argument("--logdir", type=str, default=os.path.join(work_directory, "logs"), help="Path, where logs will be stored")
    # 解析命令行参数
    args = parser.parse_args()

    # 根据评估模式设置默认配置文件路径
    if args.config is None:
        if args.eval:
            args.config = os.path.join(work_directory, "experiments/human36m/eval/human36m_eval_ex.yaml")
        else:
            args.config = os.path.join(work_directory, "experiments/human36m/train/human36m_train_ex.yaml")

    # 返回解析后的命令行参数
    return args
