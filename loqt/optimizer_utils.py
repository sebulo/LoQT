import numpy as np

import torch

import torch.utils.data

import torch.nn as nn
import transformers

import bitsandbytes as bnb
from optimizers import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
from peft_pretraining import training_utils

def classify_galore_parameters(model):
    """Classify parameters for GaLore optimizer based on module names."""
    galore_params = []
    target_modules_list = ["attn", "mlp"]
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if not any(target_key in module_name for target_key in target_modules_list):
            continue

        galore_params.append(module.weight)
    
    regular_params = [p for p in model.parameters() if id(p) not in [id(gp) for gp in galore_params]]
    return regular_params, galore_params

def create_galore_optimizer(param_groups, args, optimizer_type):
    if optimizer_type.lower() == "galore_adamw":
        return GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_type.lower() == "galore_adamw8bit":
        return GaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay, is_paged=args.is_paged)
    elif optimizer_type.lower() == "galore_adafactor":
        return GaLoreAdafactor(
            param_groups, lr=args.lr, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8,
            beta1=args.beta1, weight_decay=args.weight_decay, relative_step=False, scale_parameter=False, warmup_init=False
        )
    else:
        raise ValueError(f"Unsupported GaLore optimizer type: {optimizer_type}")



def create_layer_wise_optimizer(model, args, param_ids, update_steps=[]):
    # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
    optimizer_dict = {}
    scheduler_dict = {}
    
    # Determine the total number of scheduler steps
    total_scheduler_steps = args.num_training_steps if args.scheduler_effective_training_steps <= 0 else args.scheduler_effective_training_steps
    # Adjust for potential doubling needed for GaLore specific logic
    total_scheduler_steps *= 2
    warmup_steps = args.warmup_steps * 2

    # Determine 'update_proj_gap_arr' based on 'proj_gap_progression'
    update_proj_gap_arr = update_steps if args.proj_gap_progression != 'static' else []

    for p in model.parameters():
        if p.requires_grad:
            if args.optimizer.lower() == 'galore_adamw8bit_per_layer' and id(p) in param_ids:
                optimizer_dict[p] = GaLoreAdamW8bit(
                    [{'params': [p], 'rank': args.rank, 'update_proj_gap': args.update_proj_gap * 2, 'scale': args.galore_scale, 'proj_type': args.proj_type, 'update_proj_gap_arr': update_proj_gap_arr}],
                    lr=args.lr, weight_decay=args.weight_decay
                )
            else:
                optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

            scheduler_dict[p] = training_utils.get_scheduler(
                optimizer=optimizer_dict[p],
                scheduler_type=args.scheduler,
                num_training_steps=total_scheduler_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=args.min_lr_ratio,
            )

    return optimizer_dict, scheduler_dict

    
def create_optimizer(params, args, optimizer_type):
    if optimizer_type.lower() in ["adam", "adamw", "sgd", "adafactor", "adam8bit"]:
        return create_standard_optimizer(params, args, optimizer_type)
    elif optimizer_type.lower() in ["galore_adamw", "galore_adafactor", "galore_adamw8bit"]:
        return create_galore_optimizer(params, args, optimizer_type)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_standard_optimizer(params, args, optimizer_type):
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, eps=args.adamw_eps)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
    elif optimizer_type.lower() == "adafactor":
        return transformers.optimization.Adafactor(
            params, lr=args.lr, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8,
            beta1=args.beta1, weight_decay=args.weight_decay, relative_step=False, scale_parameter=False, warmup_init=False
        )
    elif optimizer_type.lower() == "adam8bit":
        return bnb.optim.Adam8bit(params, lr=args.lr, weight_decay=args.weight_decay, is_paged=args.is_paged)


def register_hooks(optimizer_dict, scheduler_dict):
    def optimizer_hook(p):
        if p.grad is None: 
            return
        optimizer_dict[p].step()
        optimizer_dict[p].zero_grad()
        scheduler_dict[p].step()

    for p in optimizer_dict:
        p.register_post_accumulate_grad_hook(optimizer_hook)
        
        
def setup_layerwise_optimizer(model, args, galore_params=None, update_steps=None):
    if args.optimizer.lower() not in ['adamw8bit_per_layer', 'galore_adamw8bit_per_layer']:
        return None, None, False
    
    if args.optimizer.lower() == 'adamw8bit_per_layer':
        # Use all trainable parameters for pLoRA specific setup
        optimizer_dict, scheduler_dict = create_layer_wise_optimizer(
            model, args, [id(p) for p in model.parameters() if p.requires_grad]
        )
    else:  # 'galore_adamw8bit_per_layer'
        if not galore_params:  # Ensure GaLore parameters are defined
            raise ValueError("Layer-wise optimizer for GaLore specified but no GaLore parameters classified.")
        id_galore_params = [id(p) for p in galore_params]
        update_proj_gap_arr = update_steps if args.proj_gap_progression != 'static' else []
        optimizer_dict, scheduler_dict = create_layer_wise_optimizer(
            model, args, id_galore_params, update_proj_gap_arr
        )
    
    register_hooks(optimizer_dict, scheduler_dict)
    return optimizer_dict, scheduler_dict, True
