import os
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

def load_model(model_id_or_path, device_map, load_in_4bit, torch_dtype):
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