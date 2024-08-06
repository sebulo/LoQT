import os
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM


llama_version = 7
# Specify the directory where you want to save the model and tokenizer
save_dir = f"checkpoints/llama{llama_version}b"

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Create a temporary directory for the Hugging Face cache
temp_cache_dir = tempfile.mkdtemp()

# Download the tokenizer and model using the temporary cache directory
print("Downloading the tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-2-{llama_version}b-hf", cache_dir=temp_cache_dir)
model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-2-{llama_version}b-hf", cache_dir=temp_cache_dir)

# Save the tokenizer and model to the specified directory
print(f"Saving the tokenizer and model to {save_dir}...")
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

# Verify that the model and tokenizer can be loaded from the saved directory
print("Loading the tokenizer and model from the saved directory...")
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)

print("Model and tokenizer loaded successfully from the saved directory.")