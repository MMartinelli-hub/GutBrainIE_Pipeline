# helper.py
"""
This file contains helper functions and classes
"""

import yaml

def print_structure(data, indent=0):
    """
    Recursively prints all fields and subfields of the given data

    Parameters:
    data (dict): The dictionary struccture to print
    indent (int): The number of spaces to use for indentation
    """
    if isinstance(data, dict):
        for key, value in data.items():
            print("  " * indent + f"Key: {key}")
            print_structure(value, indent + 1)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print("  " * indent + f"Item {i}:")
            print_structure(item, indent + 1)
    else:
        print("  " * indent + f"Value: {data}")


def load_config(config_path='config/config.yaml'):
    """
    Load the YAML configuration file passed as argument

    Parameters:
    config_path (str): The path to the YAML configuration file to load
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)