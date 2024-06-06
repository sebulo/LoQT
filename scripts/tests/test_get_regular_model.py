import torch
from transformers import AutoTokenizer
import os
import sys

# Adjust the path to loqt
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path, '../../'))
from loqt.LoQT import LoQTModel

def main():
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=256)

    # Checkpoint to test
    checkpoint = 'CHECKPOINT_PATH' # INSERT YOU CHECKPOINT HERE
    
    try:
        # Load the LoQT model from checkpoint
        loqt_model = LoQTModel.from_pretrained(checkpoint, device, saved_as_full_model=False)
        print(f'Model loaded from {checkpoint}')
    except Exception as e:
        print(f'Error loading model from {checkpoint}: {e}')

    # Prepare some input data
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Evaluate the model with quantized layers
    loqt_model.eval()
    with torch.no_grad():
        output_loqt = loqt_model(**inputs)

    # Get the regular model with original layers
    regular_model = loqt_model.return_original_model()
    regular_model = regular_model.to(device)
    regular_model.eval()
    with torch.no_grad():
        output_original = regular_model(**inputs)
    
    print("Mean difference in logits", torch.mean(torch.abs(output_loqt.logits - output_original.logits)))
    print('Logits are close', torch.allclose(output_loqt.logits, output_original.logits, atol=1e-1))
    assert torch.allclose(output_loqt.logits, output_original.logits, atol=1e-1), "Outputs are not close!"

    # Get tokens for both outputs
    output_loqt_tokens = tokenizer.decode(torch.argmax(output_loqt.logits, dim=-1)[0])
    output_original_tokens = tokenizer.decode(torch.argmax(output_original.logits, dim=-1)[0])

    # Compare the outputs
    tokens_match = output_loqt_tokens == output_original_tokens
    print(f'Output tokens match: {tokens_match}')
    
    assert tokens_match, "Output tokens are not the same"

    print("Looks good boss! :fire:")

if __name__ == "__main__":
    print("Starting test script")
    main()