import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def average_models(model1_path, model2_path, output_dir, alpha=0.5, device=None):
    """
    Average two Huggingface models of the same type.
    
    Args:
        model1_path (str): Path to the first model
        model2_path (str): Path to the second model
        output_dir (str): Path to save the averaged model
        alpha (float): Weight for the first model (1-alpha for the second model)
        device (str): Device to load models on ('cpu', 'cuda', etc.)
    """
    print(f"Loading model 1 from {model1_path}...")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the first model and its tokenizer
    model1 = AutoModelForCausalLM.from_pretrained(
        model1_path,
        torch_dtype=torch.float32,  # Use float32 for averaging
        device_map=device
    )
    tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
    
    # Load the second model with the same architecture
    print(f"Loading model 2 from {model2_path}...")
    model2 = AutoModelForCausalLM.from_pretrained(
        model2_path,
        torch_dtype=torch.float32,
        device_map=device
    )
    
    # Verify models have the same architecture
    model1_config = model1.config
    model2_config = model2.config
    
    # Check if model architectures match
    if model1_config.model_type != model2_config.model_type:
        raise ValueError(f"Model types don't match: {model1_config.model_type} vs {model2_config.model_type}")
    
    if model1_config.vocab_size != model2_config.vocab_size:
        raise ValueError(f"Vocabulary sizes don't match: {model1_config.vocab_size} vs {model2_config.vocab_size}")
    
    # Calculate parameter count to verify they match
    model1_params = sum(p.numel() for p in model1.parameters())
    model2_params = sum(p.numel() for p in model2.parameters())
    
    if model1_params != model2_params:
        raise ValueError(f"Parameter counts don't match: {model1_params} vs {model2_params}")
    
    print(f"Both models have matching architecture with {model1_params} parameters.")
    print(f"Averaging with alpha={alpha} (weight for model 1)")
    
    # Create a new model with the averaged weights
    avg_model = AutoModelForCausalLM.from_pretrained(
        model1_path,
        torch_dtype=torch.float32,
        device_map=device
    )
    
    # Average the model weights
    with torch.no_grad():
        for name, param in avg_model.named_parameters():
            model1_param = model1.get_parameter(name)
            model2_param = model2.get_parameter(name)
            param.data = alpha * model1_param.data + (1 - alpha) * model2_param.data
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the averaged model and tokenizer
    print(f"Saving averaged model to {output_dir}...")
    avg_model.save_pretrained(output_dir)
    tokenizer1.save_pretrained(output_dir)
    
    print("Model averaging complete!")
    return avg_model, tokenizer1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average two Huggingface models of the same type")
    parser.add_argument("--model1", type=str, required=True, help="Path to the first model")
    parser.add_argument("--model2", type=str, required=True, help="Path to the second model")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the averaged model")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for the first model (default: 0.5)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    average_models(args.model1, args.model2, args.output_dir, args.alpha, args.device)