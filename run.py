import os
import re
import yaml
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import gradio as gr

# Load model and processor
torch.cuda.empty_cache()
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.uint8,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_threshold=6.0,
)
processor = AutoProcessor.from_pretrained(model_id)

print("Model loaded!")

# Function to load characters from YAML
def load_characters(characters_folder="characters"):
    characters = {}
    for file in os.listdir(characters_folder):
        if file.endswith(".yaml"):
            with open(os.path.join(characters_folder, file), "r") as f:
                character_data = yaml.safe_load(f)
                characters[character_data["name"]] = {
                    "greeting": character_data.get("greeting", ""),
                    "context": character_data.get("context", ""),
                    "image": os.path.join(characters_folder, f"{character_data['name']}.png"),
                }
    return characters

# Load characters
characters = load_characters()
character_names = list(characters.keys())

# Chat function
def chat_step(user_input, uploaded_image, selected_character, chat_history):
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
    model.gradient_checkpointing_enable()
    output = model.generate(**inputs, max_new_tokens=256)
    model_response = processor.decode(output[0])

    # Extract the last assistant response
    matches = re.findall(r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", model_response, re.DOTALL)
    clean_response = matches[-1].strip() if matches else model_response.strip()

    # Update chat history
    chat_history.append((user_input if not uploaded_image else "Image Sent", clean_response))

    return chat_history, gr.update(value=None), gr.update(value=None)

# Save new character details
def save_new_character(name, greeting, context, character_image):
    # Create character data
    new_character = {
        "name": name,
        "greeting": greeting,
        "context": context,
    }

    # Save YAML file
    characters_folder = "characters"
    os.makedirs(characters_folder, exist_ok=True)
    
    character_data_file = os.path.join(characters_folder, f"{name}.yaml")
    with open(character_data_file, "w") as f:
        yaml.dump(new_character, f)

    # Save character image
    image_path = os.path.join(characters_folder, f"{name}.png")
    character_image.save(image_path)

    # Reload characters after saving new character
    global characters
    characters = load_characters()
    return f"Character {name} has been saved successfully!", character_image

# Update character details
def update_character_details(selected_character):
    character = characters[selected_character]
    return character["image"], character["context"]

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

    with gr.Tab("Create New Character"):
        with gr.Row():
            gr.Markdown("### Add a New Character")
            name_input = gr.Textbox(placeholder="Enter character name", label="Character Name")
            greeting_input = gr.Textbox(placeholder="Enter greeting", label="Greeting")
            context_input = gr.Textbox(placeholder="Enter context", label="Context", lines=4)
            new_character_image = gr.Image(label="Upload Character Image", type="pil")
            save_button = gr.Button("Save New Character")
            save_message = gr.Textbox(label="Save Status", interactive=False)

        # Interaction for creating a new character
        save_button.click(
            save_new_character,
            inputs=[name_input, greeting_input, context_input, new_character_image],
            outputs=[save_message, new_character_image]
        )

# Launch the Gradio app
demo.launch()
