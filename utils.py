from omegaconf import OmegaConf
import json

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)  

def load_config(file_path):
    """
    Load a YAML configuration file using OmegaConf.
    """
    config = OmegaConf.load(file_path)
    return config