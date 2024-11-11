import torch
import torch.nn as nn
import os
import torch.distributed as dist
import json


def get_proj_update_steps(args):
    steps = [0]
    update_proj_gap = args.update_proj_gap 
    current_step = update_proj_gap # Update the projection gap every update_proj_gap steps - use gradient accumulation step as the unit
    current_gap = current_step
    step_count = 0
    if args.proj_gap_progression == "static":
        while current_step <= args.num_training_steps:
            steps.append(current_step)
            current_step += update_proj_gap
    elif args.proj_gap_progression == "linear":
        increment = args.increment_size
        while current_step <= args.num_training_steps:
            steps.append(current_step)
            current_gap += increment
            step_count += 1
            current_step += int(current_gap)
    elif args.proj_gap_progression == "exponential":
        while current_step <= args.num_training_steps:
            steps.append(current_step)
            step_count += 1
            current_step += update_proj_gap + int((args.increment_size)**step_count)
                
    if args.max_proj_gap != 0:
        steps = [min(step, args.max_proj_gap) for step in steps]

    return steps
        

def get_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model

    
def create_zero_initialized_linear_layer(input_features, output_features, use_bias, device, dtype=None):
    # Create a linear layer with specified input and output features and bias setting
    linear_layer = nn.Linear(input_features, output_features, bias=use_bias)

    # Initialize the weights and biases of the layer to zero
    nn.init.constant_(linear_layer.weight, 0)
    if use_bias:
        nn.init.constant_(linear_layer.bias, 0)

    # Move the layer to the specified device and convert to the specified data type
    return linear_layer.to(device).to(dtype=dtype)

    
def compare_parameters(model):
    for name, param in model.named_parameters():
        # Gather parameters from all processes to process 0
        gathered_param = [torch.zeros_like(param) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_param, param)

        if dist.get_rank() == 0:  # Let's do the comparison on process 0
            reference = gathered_param[0]
            for rank, tensor in enumerate(gathered_param):
                if not torch.equal(reference, tensor):
                    print(f"Mismatch found in parameter {name} between rank 0 and rank {rank}")


def broadcast_parameters(model, rank, root=0):
    try:
        for param in model.parameters():
            if param.device != torch.device(f"cuda:{rank}"):
                param.data = param.data.contiguous().to(f"cuda:{rank}")
            else:
                param.data = param.data.contiguous()
            dist.broadcast(param.data, src=root)
        dist.barrier()
    except Exception as e:
        print(f"Rank {rank} encountered an error: {e}")

def log_tensor_statistics(param, name, rank):
    mean_val = torch.mean(param.data).item()
    max_val = torch.max(param.data).item()
    min_val = torch.min(param.data).item()
    std_val = torch.std(param.data).item()
    print(f"Rank {rank}, Tensor {name} - Mean: {mean_val}, Max: {max_val}, Min: {min_val}, Std: {std_val}")
    
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = torch.mean(param.grad).item()
            if grad_mean == 0:
                print(f"Gradient for {name} is zero!")
            else:
                print(f"Gradient for {name} is non-zero and mean is {grad_mean}")
        else:
            print(f"No gradient for {name}")

def eigenH_decomposition(A, out = 'u', checks=False):
    out = out.lower()
    if out == 'u':
        symmetric_input = A @ A.T  # Form A * A^H to make it symmetric
    elif out == 'v':
        symmetric_input = A.T @ A
    else:
        raise ValueError("Invalid output type. Choose 'u' or 'v'.")
    res = torch.linalg.eigh(symmetric_input)
    
    # ascending to descending
    eigenvectors = res.eigenvectors.flip(1)

    if checks:
        diff_unitary = torch.norm(eigenvectors @ eigenvectors.mH - torch.eye(A.shape[0], device=A.device), 'fro')
        print('Difference from unitary:', diff_unitary)

    return eigenvectors #, eigenvalues

def load_model_from_checkpoint(directory, model):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    files = os.listdir(directory)
    bin_file = "pytorch_model.bin"
    safetensors_file = "model.safetensors"

    checkpoint_path = None
    if bin_file in files:
        checkpoint_path = os.path.join(directory, bin_file)
        model = torch.load(checkpoint_path)
        print(f"Loaded model from {checkpoint_path}")
    elif safetensors_file in files:
        print(model)
        checkpoint_path = os.path.join(directory, safetensors_file)
        # Ensure the path is absolute
        abs_path = os.path.abspath(checkpoint_path)
        print('checkpoint_path',checkpoint_path)
        print('abs_path',abs_path)
        model = model.from_pretrained(directory)
        print(f"Loaded model from {abs_path}")
    else:
        raise FileNotFoundError("No compatible checkpoint file found (.bin or .safetensors).")

    return model


def filter_target_modules(model, target_modules_list):
    model_modules = dict(model.named_modules())
    filtered_modules = [module for module in target_modules_list if any(module in name for name in model_modules)]
    return filtered_modules


def filter_linear_target_modules(model, target_modules_list):
    filtered_modules = []
    for name, module in model.named_modules():
        if any(target in name for target in target_modules_list) and isinstance(module, nn.Linear):
            filtered_modules.append(name)
    return filtered_modules


