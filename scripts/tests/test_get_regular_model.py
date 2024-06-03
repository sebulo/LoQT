import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import os
import sys
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path, '../../'))  # Adjust the path to loqt
from loqt.LoQT import LoQTModel

def compare_model_outputs(model, tokenizer, input_text, device='cpu', update_step=0):
    """
    Compare the outputs of a model with dequantized layers against the regular model.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to preprocess the input text.
        input_text (str): The input text to be tokenized and fed to the model.
        device (str): The device to run the model on ('cpu' or 'cuda').
        save_and_load (bool): Whether to save and immediately load the checkpoint.
        args: Arguments required for saving the checkpoint.
        save_logger: Logger required for saving the checkpoint.
        update_step (int): The current update step for logging purposes.

    Returns:
        dict: A dictionary containing the differences and outputs.
    """
    # Tokenize and prepare inputs
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    breakpoint()
    # Evaluate the model with quantized layers
    model.eval()
    with torch.no_grad():
        output_loqt = model(**inputs)

    # Get the regular model with dequantized layers
    regular_model = model.return_regular_model()
    regular_model = regular_model.to(device)
    regular_model.eval()
    with torch.no_grad():
        output_dequantized = regular_model(**inputs)

    # Compare the outputs
    detailed_diff = (output_loqt.logits - output_dequantized.logits).abs()
    diff_mean = detailed_diff.mean().item()
    zero_proportion = (detailed_diff == 0).sum().item() / detailed_diff.numel()

    # Prepare the results
    results = {
        "difference_mean": diff_mean,
        "proportion_of_zeros": zero_proportion,
        "detailed_difference": detailed_diff,
        "output_loqt": output_loqt,
        "output_dequantized": output_dequantized,
        "output_loqt_logits": output_loqt.logits,
        "output_dequantized_logits": output_dequantized.logits
    }

    # Print results
    if diff_mean > 0:
        if update_step != 0:
            print("update_step:", update_step)
            
        print("Difference mean:", diff_mean)
        print('Proportion of zeros:', zero_proportion)

    model.train()
    return results


def main():
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer and the LLaMA model configuration
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=256)

    #continue_from_checkpoint = 'checkpoints/60m_LoQT_1716997317/latest_checkpoint'
    continue_from_checkpoint = 'checkpoints/60m_LoQT_1717418925/latest_checkpoint'
    loqt_model = LoQTModel.from_pretrained(continue_from_checkpoint, device)
    breakpoint()
    # Prepare some input data
    input_text = "The quick brown fox jumps over the lazy dog."
    outputs = compare_model_outputs(loqt_model, tokenizer, input_text, device)
    
    # Check if the outputs are close
    output_loqt = outputs["output_loqt"]
    output_dequantized = outputs["output_dequantized"]
    assert torch.allclose(output_loqt.logits, output_dequantized.logits, atol=1e-3), "Outputs are not close!"

    print("Looks good boss! :fire:")

if __name__ == "__main__":
    print("Starting script")
    main()


