import os
import yaml
import torch
import numpy as np
from PIL import Image
from typing import Dict

def load_config(path: str) -> Dict:
    """loads a single yml file

    Args:
        path (str): path to yml file

    Returns:
        Dict: yml dict
    """
    with open(path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
    return params

def load_state_dict_from_ckpt(ckpt_path: str) -> Dict:
    """laods a state dict from ckpt file

    Args:
        ckpt_path (str): ckpt path

    Returns:
        Dict: model state dict (weights)
    """
    device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')
    ckpt_state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    state_dict = {}
    for key, weights in ckpt_state_dict.items():
        l = key.replace("model.", "")
        state_dict[l] = weights
    return state_dict

def read_binary(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        raise ValueError(f"The path {file_path} does not exist")
    image = Image.open(file_path).convert("L")
    image = np.array(image, dtype=np.float32) / 255
    return image

def read_gray(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        raise ValueError(f"The path {file_path} does not exist")
    
    image = Image.open(file_path).convert("L")
    image = np.array(image, dtype=np.float32)
    return image

def read_rgb(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        raise ValueError(f"The path {file_path} does not exist")

    image = Image.open(file_path).convert("RGB")
    return image
    