# Visual Text Generation Web UI

This repository provides a **Visual Text Generation Web UI** for interacting with a large pre-trained model, specifically designed for generating responses based on text and image inputs. The UI supports character-based interactions and allows users to chat with different characters.

## Features

- **Interactive Chat Interface**: Users can input text and images for natural language processing and generation.
- **Character Support**: You can customize and define new characters, complete with persona, greetings, and context. Each character can also have an associated image for a visual experience.
- **Visual Inference**: Supports the ability to input images along with text prompts, allowing for more interactive responses.
- **Customizable Persona**: Characters have customizable personalities and greetings, creating an engaging user experience.

## Requirements

- Python 3.8+
- PyTorch (with GPU support)
- Gradio
- Transformers library

### Install Dependencies

You can install the required Python dependencies by running:

```bash
pip install -r requirements.txt
```

### Setup

1. Clone the repository

```bash
git clone https://github.com/ricardo2001l/visual-text-generation-webui.git
cd visual-text-generation-webui
```

2. Run the Web UI: Start the web interface by running:

3. Set the Model: On the model tab you can set the model [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/) can be used is used for text generation. It will be downloaded on the models folder. Also you can also choose a finetuned model based on llama 3.2 11B. You may be could also use another Vision model of this kind.

```bash
python run.py
```
This will start a local web server at http://localhost:7860, where you can interact with the model and characters.


### Contributing

Feel free to fork the repository and submit pull requests. If you'd like to add more features or fix any bugs, contributions are always welcome!

1. Fork the repository
2. Create your branch (git checkout -b feature-branch)
3. Commit your changes (git commit -am 'Add new feature')
3. Push to the branch (git push origin feature-branch)
4. Create a new Pull Request
