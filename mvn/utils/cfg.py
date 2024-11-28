import yaml
from easydict import EasyDict as edict


def load_config(path):
    with open(path) as f:
        config = edict(yaml.safe_load(f))

    return config