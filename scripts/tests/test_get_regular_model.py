import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import os
import sys
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path, '../../'))  # Adjust the path to loqt
from loqt.LoQT import LoQTModel


def main():
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer and the LLaMA model configuration
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=256)

    continue_from_checkpoint = 'checkpoints/60m_LoQT_1716997317/latest_checkpoint'
    loqt_model = LoQTModel.from_pretrained(continue_from_checkpoint, device)
    # checkpoints/60m_LoQT_1716992064/latest_checkpoint

    # Prepare some input data
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Perform a forward pass with the LoQT model
    print('loqt_model',loqt_model)
    
    loqt_model = loqt_model.to(device)
    loqt_model.eval()
    with torch.no_grad():
        output_loqt = loqt_model(**inputs)

    # Call return_regular_model to get the model with only the dequantized layers
    regular_model = loqt_model.return_regular_model()
    print('loqt_model', regular_model)
    # Perform a forward pass with the dequantized model
    regular_model = regular_model.to(device)
    regular_model.eval()
    with torch.no_grad():
        output_dequantized = regular_model(**inputs)

    # Compare the outputs
    diff = (output_loqt.logits - output_dequantized.logits).abs().mean().item()
    print("Difference mean:", diff)
    diff = (output_loqt.logits - output_dequantized.logits).abs()
    print('proportion of zeros:', (diff == 0).sum().item() / diff.numel())
    print("Difference:", diff)
    print('output_loqt.logits',output_loqt.logits)
    print('output_dequantized.logits',output_dequantized.logits)

    # Check if the outputs are close
    assert torch.allclose(output_loqt.logits, output_dequantized.logits, atol=1e-3), "Outputs are not close!"

    print("Looks good boss! :fire:")

if __name__ == "__main__":
    print("Starting script")
    main()


