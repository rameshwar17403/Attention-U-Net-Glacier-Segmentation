import os
import random
import numpy as np
import torch
import yaml

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_config(config_path="configs/default.yaml"):
    if not os.path.exists(config_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        alt_path = os.path.join(repo_root, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
