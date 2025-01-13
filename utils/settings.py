import os
import yaml

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