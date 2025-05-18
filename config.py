import os
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')

with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
