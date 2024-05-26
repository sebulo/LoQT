import torch
from loqt.LoQT import LoQTModel

def optimizer_memory_usage_in_MB(optimizer):
    """
    Calculates the memory usage of a PyTorch optimizer's state dict in megabytes (MB).
    """
    optimizer_state = optimizer.state_dict()
    total_size_bytes = 0
    for state in optimizer_state['state'].values():
        for k, v in state.items():
            if torch.is_tensor(v):
                total_size_bytes += v.numel() * v.element_size()
            elif isinstance(v, list):
                for item in v:
                    if torch.is_tensor(item):
                        total_size_bytes += item.numel() * item.element_size()
    memory_usage_MB = total_size_bytes / (1024 ** 2)  # Convert bytes to MB
    gradient_memory = sum(p.numel()*p.element_size() for group in optimizer.param_groups for p in group['params'] if p.requires_grad)/(1024**2)
    return memory_usage_MB, gradient_memory

def layer_wise_memory_usage_in_MB(optimizer_dict, scheduler_dict):
    memory = 0
    for val in optimizer_dict.values():
        for state in val.state_dict()['state'].values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    memory += v.numel() * v.element_size()
                elif isinstance(v, list):
                    for item in v:
                        if torch.is_tensor(item):
                            memory += item.numel() * item.element_size()
    return memory / (1024 ** 2)

def galore_optim_memory_usage_in_MB(optimizer):
    memory = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                memory += v.numel() * v.element_size()
    
    return memory / (1024 ** 2)

def model_memory_usage_in_MB(model):
    total_size_bytes = sum([p.numel() * p.element_size() for p in model.parameters()])
    gradients = sum([p.numel() * p.element_size() for p in model.parameters() if p.requires_grad])/(1024**2)
    memory_usage_MB = total_size_bytes / (1024 ** 2)
    return memory_usage_MB, gradients

def get_gpu_metrics_nvitop(this_process, suffix=''):
    # Update the GPU status
    this_process.update_gpu_status()
    
    # Retrieve GPU metrics
    gpu_metrics = {
        f'device{suffix}/memory_used_MB': f"{float(this_process.device.memory_used()) / (1 << 20):.2f} MB",  # Convert bytes to MiBs
        f'device{suffix}/memory_percent': f"{this_process.device.memory_percent():.2f} %",
        f'device{suffix}/memory_utilization': f"{this_process.device.memory_utilization():.2f} %",
        f'device{suffix}/gpu_utilization': f"{this_process.device.gpu_utilization():.2f} %",
        f'process{suffix}/cpu_percent': f"{this_process.cpu_percent():.2f} %",
        f'process{suffix}/memory_percent': f"{this_process.memory_percent():.2f} %",
        f'process{suffix}/used_gpu_memory_MB': f"{float(this_process.gpu_memory()) / (1 << 20):.2f} MB",  # Convert bytes to MiBs
        f'process{suffix}/gpu_sm_utilization': f"{this_process.gpu_sm_utilization():.2f} %",
        f'process{suffix}/gpu_memory_utilization': f"{this_process.gpu_memory_utilization():.2f} %",
    }
    
    return gpu_metrics



def log_memory_usage(model, optimizer, scheduler_dict, args, logger):
    def get_model_memory_usage(model, include_quant_state=True):
        if args.use_loqt and args.optimizer.lower() not in ["adamw8bit_per_layer"]:
            return LoQTModel.model_memory_usage_in_MB(model, include_quant_state=include_quant_state)
        elif args.optimizer.lower() in ["galore_adamw", "galore_adamw8bit"]:
            return model_memory_usage_in_MB(model)[0]  # Only the model memory usage
        elif args.optimizer.lower() in ["adamw8bit_per_layer", "galore_adamw8bit_per_layer"]:
            return model_memory_usage_in_MB(model)[0]  # Only the model memory usage
        elif args.optimizer.lower() in ['adamw', 'adam']:
            return model_memory_usage_in_MB(model)[0]  # Only the model memory usage
        else:
            raise ValueError("Unsupported optimizer type for memory logging")


    
    if args.use_loqt and args.optimizer.lower() not in ["adamw8bit_per_layer"]:
        memory_usage_optimizer, memory_usage_gradients = optimizer_memory_usage_in_MB(optimizer)
        memory_usage_model = get_model_memory_usage(model, include_quant_state=True)
    elif args.optimizer.lower() in ["galore_adamw", "galore_adamw8bit"]:
        memory_usage_optimizer = galore_optim_memory_usage_in_MB(optimizer)
        memory_usage_model = get_model_memory_usage(model)
        memory_usage_gradients = 0  # Assuming galore optimizers do not use gradients memory
    elif args.optimizer.lower() in ["adamw8bit_per_layer", "galore_adamw8bit_per_layer"]:
        memory_usage_optimizer = layer_wise_memory_usage_in_MB(optimizer, scheduler_dict)
        memory_usage_model = get_model_memory_usage(model)
        memory_usage_gradients = 0  # Assuming layer-wise optimizers do not use gradients memory
    elif args.optimizer.lower() in ['adamw', 'adam']:
        memory_usage_optimizer, memory_usage_gradients = optimizer_memory_usage_in_MB(optimizer)
        memory_usage_model = get_model_memory_usage(model)
    else:
        raise ValueError("Unsupported optimizer type for memory logging")
    
    optmizer_and_model_memory = memory_usage_optimizer + memory_usage_model
    
    logger.info(f"Memory usage of model: {memory_usage_model} MB")
    logger.info(f"Memory usage of optimizer: {memory_usage_optimizer} MB")
    logger.info(f"Memory usage of optimizer and model: {optmizer_and_model_memory} MB")
    
    memory_usage_dict = {
        "model": memory_usage_model,
        "optimizer": memory_usage_optimizer,
        "optimizer_and_model": optmizer_and_model_memory,
        "gradients": memory_usage_gradients
    }
    return memory_usage_dict
        
