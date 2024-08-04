import torch
from transformers import AutoTokenizer
import os
import sys
import subprocess

# Adjust the path to loqt
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path, '../../'))
from loqt.LoQT import LoQTModel

def run_training():
    # Run training script
    training_command = [
        "torchrun", "--standalone", "--nproc_per_node", "1", "--nnodes", "1", "torchrun_main.py",
        "--model_config", "configs/llama_60m.json",
        "--seed", "42",
        "--lr", "0.01",
        "--rank", "128",
        "--lora_alpha", "0.4",
        "--update_proj_gap", "50",
        "--batch_size", "256",
        "--total_batch_size", "512",
        "--num_training_steps", "100",
        "--warmup_steps", "10",
        "--eval_every", "0",
        "--save_every", "50",
        "--dtype", "bfloat16",
        "--optimizer", "adamw",
        "--use_loqt", "True",
        "--quantize_w", "4bit",
        "--quantize_projection_matrix", "4bit",
        "--compensate_quant_error_iterations", "5",
        "--proj_gap_progression", "exponential",
        "--increment_size", "1.2",
        "--save_original_model", "True",
        "--run_final_eval", "False",
        "--save_dir", "tmp",
        "--name", "60m_LoQT"
    ]
    subprocess.run(training_command, check=True)
def get_latest_checkpoint_path(checkpoint_dir):
        
        # Ensure the checkpoint directory exists
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"The directory '{checkpoint_dir}' does not exist.")
        print(f"Checkpoint directory exists: {checkpoint_dir}")

        # Get the first folder in the checkpoint directory
        subfolders = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
        if not subfolders:
            raise FileNotFoundError("No subdirectories found in the 'tmp' directory.")
        checkpoint_subfolder = subfolders[0]

        # Construct the path to the latest checkpoint directory
        latest_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_subfolder, 'latest_checkpoint')
        print(f"Constructed path to latest checkpoint directory: {latest_checkpoint_path}")
        
        return latest_checkpoint_path
    
def run_tests():
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=256)

    # Define the directory to search for the checkpoint
    checkpoint_dir = "tmp"
    latest_checkpoint_path = get_latest_checkpoint_path(checkpoint_dir)

    try:
        # Load the LoQT model from checkpoint
        loqt_model = LoQTModel.from_pretrained(latest_checkpoint_path, device, saved_as_full_model=False)
        print(f'Model loaded from {latest_checkpoint_path}')
    except Exception as e:
        print(f'Error loading model from {latest_checkpoint_path}: {e}')

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
    print("Starting training script")
    try:
        run_training()
        print("Finished")
    except Exception as e:
        print(f"Error during training: {e}")
    print("Training complete. Starting test script")
    run_tests()
    # Clean up tmp directory
    try:
        subprocess.run(["rm", "-rf", "tmp"], check=True)
    except Exception as e:
        print(f"Error during cleanup: {e}")
    print("Cleanup complete")
    
