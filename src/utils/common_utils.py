import os
from box.exceptions import BoxValueError
import yaml
from box import ConfigBox
from pathlib import Path
import os

def read_yaml(path_to_yaml) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

def get_weights_file_path(config, epoch: str):
    model_folder = config.model_folder
    model_basename = config.model_basename
    model_filename = f"{model_basename}{epoch}.pth"

    full_path = os.path.join(model_folder, model_filename)

    return full_path