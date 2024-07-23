import torch
import yaml
import numpy as np
from typing import Literal
from pathlib import Path
from yourdfpy import URDF
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def get_config(save_or_not=False, print_or_not=False):
    config = {}  # update config

    if print_or_not:
        print_config(config, num_char=100)
    if save_or_not:
        save_config(config)
    return config


def save_config(config):
    with open(config["WORK_DIR"] + "/config.yaml", "w") as f:
        for value in config.keys():
            if not isinstance(config[value], dict):
                f.write(f"{value}\t{config[value]}\n")
            else:
                f.write(f"{value}:\n")
                for v in config[value].keys():
                    f.write(f"\t{v}\t{config[value][v]}\n")
            f.write("\n")


def save_config_given_dir(config, dir):
    with open(dir + "/config.yaml", "w") as f:
        for value in config.keys():
            if not isinstance(config[value], dict):
                f.write(f"{value}:\t{config[value]}\n")
            else:
                for new_values in config[value].keys():
                    if not isinstance(config[value][new_values], dict):
                        f.write(f"{value}:\t{config[value][new_values]}\n")
                    else:
                        f.write(f"{value}:\n")
                        for v in config[value][new_values].keys():
                            f.write(f"\t{v}\t{config[value][new_values][v]}\n")
            f.write("\n")


def print_config(config, num_char):
    print("=" * num_char)
    print("Configuration:")
    print("=" * num_char)

    for value in config.keys():
        if not isinstance(config[value], dict):
            print(f"{value}\t{config[value]}", flush=True)
        else:
            print(f"{value}:")
            for v in config[value].keys():
                print(f"\t{v}\t{config[value][v]}", flush=True)

    print("=" * num_char)
    print()
