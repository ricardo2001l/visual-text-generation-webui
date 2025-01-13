import gradio as gr
import os
from utils import (
    load_settings,
    save_settings,
    load_model,
    load_characters,
    save_new_character,
    chat_step,
    parse_image_metadata,
)

# Load settings
settings = load_settings()

# Load characters
characters = load_characters(settings["characters_folder"])
character_names = list(characters.keys())

# Function to list available models
def list_models():
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return [name for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name))]

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
        model_id_input = gr.Textbox(placeholder="Enter model ID or local path", label="Model HuggingFace ID (example: meta-llama/Llama-3.2-11B-Vision-Instruct)")
        model_dropdown = gr.Dropdown(choices=list_models(), label="Available Models", interactive=True)
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

        # Update the model ID input when a model is selected from the dropdown
        model_dropdown.change(
            lambda model_name: os.path.join("models", model_name),
            inputs=model_dropdown,
            outputs=model_id_input
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
            lambda selected: (characters[selected]["image"], characters[selected]["context"]),
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
            lambda: ([], None, None, []),
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
            parse_image_metadata,
            inputs=new_character_image,
            outputs=[name_input, context_input, greeting_input]
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
            lambda device_map, load_in_4bit, torch_dtype, characters_folder, max_new_tokens, temperature: (
                save_settings({
                    "device_map": device_map,
                    "load_in_4bit": load_in_4bit,
                    "torch_dtype": torch_dtype,
                    "characters_folder": characters_folder,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                }),
                "Settings updated successfully!"
            ),
            inputs=[device_map_setting, load_in_4bit_setting, torch_dtype_setting, characters_folder_setting, max_new_tokens_setting, temperature_setting],
            outputs=save_settings_status
        )

# Launch the Gradio app
demo.launch()