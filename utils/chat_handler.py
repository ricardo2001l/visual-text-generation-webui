from PIL import Image
import re
import gradio as gr
from utils import settings

def chat_step(user_input, uploaded_image, selected_character, chat_history):
    if settings.model is None or settings.processor is None:
        print("Model or processor not loaded. Please load a model first.")
        return chat_history, gr.update(value=None), gr.update(value=None)

    # Get character data
    character = settings.characters[selected_character]
    role_message = {"role": "system", "content": character["context"]}

    # Prepare messages
    messages = [role_message]
    for msg in chat_history:
        if not isinstance(msg['content'], str):
            continue 
        messages.append(msg)

    # Handle image and user input
    if uploaded_image:
        # If an image is uploaded, process the image
        image = Image.open(uploaded_image)
        messages.append({"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_input}  # Include the text in the message
        ]})
        input_text = settings.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = settings.processor(
            images=image,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(settings.model.device)

        # Add the image and text to the chat history
        chat_history.append({"role": "user", "content": f"{user_input}"})
        chat_history.append({"role": "user", "content": gr.Image(uploaded_image)})
    else:
        # If no image is uploaded, ensure the text doesn't reference an image
        messages.append({"role": "user", "content": user_input})
        input_text = settings.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = settings.processor(
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(settings.model.device)

        # Add the text to the chat history
        chat_history.append({"role": "user", "content": user_input})

    # Generate response
    try:
        settings.model.gradient_checkpointing_enable()
        output = settings.model.generate(**inputs, max_new_tokens=settings.config["max_new_tokens"], temperature=settings.config["temperature"])
        model_response = settings.processor.decode(output[0])

        print(model_response)
        # Extract the last assistant response
        matches = re.findall(r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", model_response, re.DOTALL)
        clean_response = matches[-1].strip() if matches else model_response.strip()

        # Update chat history with the model's response
        chat_history.append({"role": "assistant", "content": clean_response})

        return chat_history, gr.update(value=None), gr.update(value=None)
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return chat_history, gr.update(value=None), gr.update(value=None)