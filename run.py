import os
import re
import yaml
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import gradio as gr
import base64
import json

# Global variables for model and processor
model = None
processor = None

# Load settings from settings.yaml
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

# Save settings to settings.yaml
def save_settings(settings):
    with open("settings.yaml", "w") as f:
        yaml.dump(settings, f)

# Load settings
settings = load_settings()

# Function to load a model
def load_model(model_id_or_path, device_map, load_in_4bit, torch_dtype):
    global model, processor
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()

        # Check if the model is a local path or a Hugging Face model ID
        if os.path.exists(model_id_or_path):
            # Load from local path
            model_path = model_id_or_path
        else:
            # Save Hugging Face models to the 'models/' folder
            model_path = os.path.join("models", model_id_or_path.replace("/", "_"))
            print(not os.path.exists(model_path))
            if not os.path.exists(model_path):
                print(f"Downloading model '{model_id_or_path}' from Hugging Face...")
                os.makedirs(model_path, exist_ok=True)

                # Download the model and processor
                model = MllamaForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path=model_id_or_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=getattr(torch, torch_dtype),
                    device_map=device_map,
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_storage=torch.uint8,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_threshold=6.0,
                )
                processor = AutoProcessor.from_pretrained(model_id_or_path)

                # Save the model and processor to the local path
                model.save_pretrained(model_path)
                processor.save_pretrained(model_path)
                print(f"Model saved to {model_path}")

        # Load the model and processor from the local path
        model = MllamaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_path,
            low_cpu_mem_usage=True,
            torch_dtype=getattr(torch, torch_dtype),
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_storage=torch.uint8,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_threshold=6.0,
        )
        processor = AutoProcessor.from_pretrained(model_path)

        return f"Model '{model_id_or_path}' loaded successfully!"
    except Exception as e:
        return f"Error loading model: {str(e)}"

# Function to load characters from YAML
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

# Load characters
characters = load_characters(settings["characters_folder"])
character_names = list(characters.keys())

# Chat function
def chat_step(user_input, uploaded_image, selected_character, chat_history):
    global model, processor
    if model is None or processor is None:
        return chat_history, "Model or processor not loaded. Please load a model first.", gr.update(value=None), gr.update(value=None)

    # Get character data
    character = characters[selected_character]
    role_message = {"role": "system", "content": character["context"]}

    # Prepare messages
    messages = [role_message]
    for user_msg, model_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": model_msg})

    # Handle image and user input
    if uploaded_image:
        # If an image is uploaded, process the image
        image = Image.open(uploaded_image)
        messages.append({"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_input}
        ]})
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            images=image,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
    else:
        # If no image is uploaded, ensure the text doesn't reference an image
        messages.append({"role": "user", "content": user_input})
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

    # Generate response
    try:
        model.gradient_checkpointing_enable()
        output = model.generate(**inputs, max_new_tokens=settings["max_new_tokens"], temperature=settings["temperature"])
        model_response = processor.decode(output[0])

        # Extract the last assistant response
        matches = re.findall(r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", model_response, re.DOTALL)
        clean_response = matches[-1].strip() if matches else model_response.strip()

        # Update chat history
        chat_history.append((user_input if not uploaded_image else "Image Sent", clean_response))

        return chat_history, gr.update(value=None), gr.update(value=None)
    except Exception as e:
        return chat_history, f"Error generating response: {str(e)}", gr.update(value=None), gr.update(value=None)

# Function to parse metadata from the image
def parse_image_metadata(img):
    try:
        # Access metadata stored in the image
        metadata = img.info

        # Extract the base64-encoded 'chara' field
        chara_encoded = metadata.get("chara", "")
        if not chara_encoded:
            print("No 'chara' field found in metadata.")
            return "", "", ""

        # Decode the base64 string
        chara_decoded = base64.b64decode(chara_encoded).decode('utf-8')

        # Parse the decoded string as JSON
        chara_data = json.loads(chara_decoded)

        # Extract the required fields
        name = chara_data.get("name", "")
        context = chara_data.get("description", "")  # Use 'description' as 'context'
        greeting = chara_data.get("first_mes", "")  # Use 'first_mes' as 'greeting'

        print(f"Name: {name}")
        print(f"Context: {context}")
        print(f"Greeting: {greeting}")

        # Return the parsed metadata as separate values
        return name, context, greeting
    except Exception as e:
        print(f"Error parsing image metadata: {e}")
        return "", "", ""  # Return empty values if there's an error

# Save new character details
def save_new_character(name, greeting, context, character_image):
    # Create character data
    new_character = {
        "name": name,
        "greeting": greeting,
        "context": context,
    }

    # Save YAML file
    characters_folder = settings["characters_folder"]
    os.makedirs(characters_folder, exist_ok=True)

    character_data_file = os.path.join(characters_folder, f"{name}.yaml")
    with open(character_data_file, "w") as f:
        yaml.dump(new_character, f)

    # Save character image
    image_path = os.path.join(characters_folder, f"{name}.png")
    character_image.save(image_path)

    # Reload characters after saving new character
    global characters
    characters = load_characters(characters_folder)
    return f"Character {name} has been saved successfully!", character_image

# Update character details
def update_character_details(selected_character):
    character = characters[selected_character]
    return character["image"], character["context"]

# Chat function with reset capability
def reset_chat():
    return [], gr.update(value=None), gr.update(value=None), []

# Function to update settings
def update_settings(device_map, load_in_4bit, torch_dtype, characters_folder, max_new_tokens, temperature):
    global settings
    settings = {
        "device_map": device_map,
        "load_in_4bit": load_in_4bit,
        "torch_dtype": torch_dtype,
        "characters_folder": characters_folder,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }
    save_settings(settings)
    return "Settings updated successfully!"

# Gradio interface
with gr.Blocks(css=".chatbox {height: 400px; overflow-y: auto;}") as demo:
    # Title
    gr.Markdown(
        """
        # Vision + Language Chat with Characters
        Select a character, chat with them, and send images!
        """,
        elem_id="title"
    )

    with gr.Tab("Load Model"):
        gr.Markdown("### Load a Custom Model")
        model_id_input = gr.Textbox(placeholder="Enter model ID or local path", label="Model HugginFace ID (example: meta-llama/Llama-3.2-11B-Vision-Instruct)")
        device_map_input = gr.Dropdown(choices=["auto", "cpu", "cuda"], value=settings["device_map"], label="Device Map")
        load_in_4bit_input = gr.Checkbox(value=settings["load_in_4bit"], label="Load in 4-bit")
        torch_dtype_input = gr.Dropdown(choices=["float16", "float32"], value=settings["torch_dtype"], label="Torch Dtype")
        load_model_button = gr.Button("Load Model")
        load_model_status = gr.Textbox(label="Model Loading Status", interactive=False)

        # Interaction for loading a model
        load_model_button.click(
            load_model,
            inputs=[model_id_input, device_map_input, load_in_4bit_input, torch_dtype_input],
            outputs=load_model_status
        )

    with gr.Tab("Chat with Character"):
        with gr.Row():
            # Character panel
            with gr.Column(scale=1):
                gr.Markdown("### Character Selection")
                selected_character = gr.Dropdown(
                    choices=character_names,
                    value=character_names[0],
                    label="Choose a Character"
                )
                character_image = gr.Image(label="Character Image", type="filepath")
                character_context = gr.Textbox(
                    label="Character Context",
                    lines=5,
                    interactive=False
                )

            # Chat panel
            with gr.Column(scale=3):
                gr.Markdown("### Chat Window")
                chatbox = gr.Chatbot(label="Chat with the Character", elem_id="chatbox")
                user_input = gr.Textbox(lines=1, placeholder="Enter your message here", label="Message")
                submit_btn = gr.Button("Send")
                upload_image = gr.Image(label="Attach Image (optional)", type="filepath")
                reset_btn = gr.Button("Reset Chat")
        # Hidden state for chat
        chat_history = gr.State([])

        # Update character image and context
        selected_character.change(
            update_character_details,
            inputs=selected_character,
            outputs=[character_image, character_context]
        )

        # Interaction for chatting
        submit_btn.click(
            chat_step,
            inputs=[user_input, upload_image, selected_character, chat_history],
            outputs=[chatbox, upload_image, user_input]
        )

        # Reset chat interaction
        reset_btn.click(
            reset_chat,
            outputs=[chatbox, upload_image, user_input, chat_history]
        )

    with gr.Tab("Create New Character"):
        with gr.Row():
            gr.Markdown("### Add a New Character")
        with gr.Row():
            with gr.Column():
                name_input = gr.Textbox(placeholder="Enter character name", label="Character Name")
                greeting_input = gr.Textbox(placeholder="Enter greeting", label="Greeting")
                context_input = gr.Textbox(placeholder="Enter context", label="Context", lines=4)
            with gr.Column():
                new_character_image = gr.Image(label="Upload Character Image", type="pil")
        with gr.Row():
            with gr.Column():
                save_button = gr.Button("Save New Character")
                save_message = gr.Textbox(label="Save Status", interactive=False)

        # Interaction for creating a new character
        save_button.click(
            save_new_character,
            inputs=[name_input, greeting_input, context_input, new_character_image],
            outputs=[save_message, new_character_image]
        )

        # Auto-fill character details from image metadata
        new_character_image.change(
            parse_image_metadata,  # Function to parse metadata
            inputs=new_character_image,  # Input is the uploaded image
            outputs=[name_input, context_input, greeting_input]  # Outputs are the three textboxes
        )

    with gr.Tab("Settings"):
        gr.Markdown("### App Settings")
        device_map_setting = gr.Dropdown(choices=["auto", "cpu", "cuda"], value=settings["device_map"], label="Device Map")
        load_in_4bit_setting = gr.Checkbox(value=settings["load_in_4bit"], label="Load in 4-bit")
        torch_dtype_setting = gr.Dropdown(choices=["float16", "float32"], value=settings["torch_dtype"], label="Torch Dtype")
        characters_folder_setting = gr.Textbox(value=settings["characters_folder"], label="Characters Folder")
        max_new_tokens_setting = gr.Number(value=settings["max_new_tokens"], label="Max New Tokens")
        temperature_setting = gr.Slider(minimum=0.1, maximum=1.0, value=settings["temperature"], label="Temperature")
        save_settings_button = gr.Button("Save Settings")
        save_settings_status = gr.Textbox(label="Save Status", interactive=False)

        # Interaction for saving settings
        save_settings_button.click(
            update_settings,
            inputs=[device_map_setting, load_in_4bit_setting, torch_dtype_setting, characters_folder_setting, max_new_tokens_setting, temperature_setting],
            outputs=save_settings_status
        )

# Launch the Gradio app
demo.launch()