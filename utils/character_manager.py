import os
import yaml

from utils import settings

def load_characters(characters_folder="characters"):
    characters = {}
    for file in os.listdir(characters_folder):
        if file.endswith(".yaml"):
            with open(os.path.join(characters_folder, file), "r") as f:
                character_data = yaml.safe_load(f)
                characters[character_data["name"]] = {
                    "greeting": character_data.get("greeting", ""),
                    "context": "You act as " + character_data["name"] + ". You are: " + character_data["name"] + " and this is your behaviour: " + character_data.get("context", ""),
                    "image": os.path.join(characters_folder, f"{character_data['name']}.png"),
                }
    return characters

def save_new_character(name, greeting, context, character_image):
    # Create character data
    new_character = {
        "name": name,
        "greeting": greeting,
        "context": context,
    }

    # Save YAML file
    os.makedirs(settings.characters_folder, exist_ok=True)
    character_data_file = os.path.join(settings.characters_folder, f"{name}.yaml")
    with open(character_data_file, "w") as f:
        yaml.dump(new_character, f)

    # Save character image
    image_path = os.path.join(settings.characters_folder, f"{name}.png")
    character_image.save(image_path)

    return f"Character {name} has been saved successfully!", character_image