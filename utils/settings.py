import os
import yaml
import gradio as gr

from utils.character_manager import load_characters


def init():
    global config, characters, characters_folder, character_names # Load settings
    config = load_settings()
    
    # Load characters# Load characters
    characters = load_characters(config["characters_folder"])
    character_names = list(characters.keys())
    characters_folder = config["characters_folder"]

    global model, processor
    model = None
    processor = None


def load_settings():
    if not os.path.exists("settings.yaml"):
        # Create default settings if the file doesn't exist
        default_settings = {
            "device_map": "auto",
            "load_in_4bit": True,
            "torch_dtype": "float16",
            "characters_folder": "characters",
            "max_new_tokens": 4096,
            "temperature": 0.7,
        }
        with open("settings.yaml", "w") as f:
            yaml.dump(default_settings, f)
        return default_settings
    else:
        with open("settings.yaml", "r") as f:
            return yaml.safe_load(f)

def save_settings(settings):
    with open("settings.yaml", "w") as f:
        yaml.dump(settings, f)