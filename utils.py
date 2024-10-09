# helper.py
"""
This file contains helper functions and classes
"""

import yaml
import json
import os

def print_structure(data, indent=0):
    """
    Recursively prints all fields and subfields of the given data.

    Parameters:
    data (dict): The dictionary struccture to print.
    indent (int): The extra number of spaces to use for indentation (default: 1).
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
    Load the YAML configuration file passed as argument.

    Parameters:
    config_path (str): The path to the YAML configuration file to load.

    Returns:
    dict: A dictionary containing the loaded configuration parameters. 
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def load_articles_from_json(filename):
    """
    Loads articles from a JSON file into a dictionary variable.
    Includes error handling for file I/O and JSON decoding
    
    Parameters:
    filename (str): The path to the JSON file containing the articles.

    Returns:
    dict: A dictionary containing the articles loaded from the JSON file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f'The file {filename} does not exist')

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f'Error decoding JSON from file {filename}: {e}')
    
    return articles
    
    
"""
RegEx to match entire line having certain keywords
^.*(word1|word2|word3).*\n?
"""