import re

def chat_step(user_input, uploaded_image, selected_character, chat_history, model, processor, max_new_tokens, temperature):
    if model is None or processor is None:
        return chat_history, "Model or processor not loaded. Please load a model first.", None, None

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
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        model_response = processor.decode(output[0])

        # Extract the last assistant response
        matches = re.findall(r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", model_response, re.DOTALL)
        clean_response = matches[-1].strip() if matches else model_response.strip()

        # Update chat history
        chat_history.append((user_input if not uploaded_image else "Image Sent", clean_response))

        return chat_history, None, None
    except Exception as e:
        return chat_history, f"Error generating response: {str(e)}", None, None