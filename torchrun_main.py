import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger
from nvitop import Device, GpuProcess


from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

from loqt.LoQT import LoQTModel
from loqt.utils import get_model, get_proj_update_steps, broadcast_parameters, load_model_from_checkpoint
from loqt.optimizer_utils import create_optimizer, setup_layerwise_optimizer, classify_galore_parameters
from loqt.memory_utils import get_gpu_metrics_nvitop, log_memory_usage

transformers.logging.set_verbosity_error()



def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--eval_at_continue_from",  default=True, type=lambda x: x.lower() == "true", help='Perform evaluation just after loading')
    parser.add_argument("--skip_batches_in_continue_from", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam", help="adam, adamw, adafactor, adam8bit, galore_adamw, galore_adafactor, galore_adamw8bit")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--adamw_eps", type=float, default=1e-6)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--save_original_model", default=False, type=lambda x: x.lower() == "true", help='Saves both the original and the model with the adapters if True')
    parser.add_argument("--save_dir", type=str, default='checkpoints')
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float16") # make fallback to fp16
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)   
    parser.add_argument("--num_eval_tokens", type=int, default=10000000)
    
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    
    parser.add_argument("--rank", type=int, default=128)
    
    # GaLore parameters
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std") # chooses the lora the smalles lora factor as the projection matrix. 'left' or 'right' or 'std'
    
    # LoQT Parameters
    parser.add_argument("--use_loqt", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--lora_alpha", type=float, default=0.5)
    parser.add_argument('--compensate_quant_error_iterations', type=int, default=0, help='Number of iterations to run the joint optimization of Lora/quant')
    parser.add_argument('--proj_gap_progression', type=str, default="static", choices=["static", "linear", "exponential"])
    parser.add_argument('--increment_size', type=float, default=1.2, help="The factor for increasing warmup steps either the linear steps or the exponential factor")
    parser.add_argument('--max_proj_gap', type=float, default=0)
    parser.add_argument("--use_eigenh_for_projection", default=False, type=lambda x: x.lower() == "true", help="If false, use SVD for projection")

    # Quantization Parameters
    parser.add_argument("--quantize_w", type=str, default=None, choices=["1bit", "4bit", "8bit"])
    parser.add_argument("--use_double_quant", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    parser.add_argument('--quantize_projection_matrix', default=None, type=str, help='4bit for 4bit quantization of projection matrix')

    # Offloading Parameters
    parser.add_argument('--use_offloading', default=False, type=lambda x: x.lower() == "true")

    # Paged Memory Parameters (for 8-bit Adam bitsandbytes implementation)
    parser.add_argument("--is_paged", default=False, type=lambda x: x.lower() == "true")

    # General Training Parameters
    parser.add_argument("--single_gpu", default=False, action="store_true", help="Disable DDP, use single GPU")
    parser.add_argument("--run_final_eval", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--dataset_path", type=str, default="./data/c4")

    # Scheduler Parameters
    parser.add_argument("--scheduler_effective_training_steps", type=int, default=0, help="Number of training steps for the scheduler")

    # Logging Parameters
    parser.add_argument("--wandb_project", type=str, default="pretraining-loqt")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument('--wandb_tag', type=str, default=None, help='Optional single tag for the wandb experiment')
    parser.add_argument('--log_max_memory', default=False, type=lambda x: x.lower() == "true")
    parser.add_argument('--log_max_memory_steps', type=int, default=1, help="Interval for logging maximum memory usage")

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size, dataset=None):
    is_training_at_entry = model.training
    model.eval()
    _time = time.time()
    if dataset is None:
        val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True)
    else:
        val_data = dataset
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = args.num_eval_tokens
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    if is_training_at_entry:
        model.train()
    return total_loss, evaluated_on_tokens


def main(args):    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"global_rank {global_rank}, local_rank {local_rank}, world_size {world_size}")
    
    # if args.use_offloading:
    #     assert world_size == 1, "Offloading is only supported for single GPU training"
    if torch.backends.mps.is_available():  # Check for MPS availability
        device = torch.device("mps")
        logger.info(f"Using MPS on Mac M1/M2, local rank {local_rank}, device: {device}")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {device}")

    dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", rank=global_rank, world_size=world_size)
    logger.info("Process group initialized")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if global_rank == 0: # only once place
        # Creating a unique directory name with model name and a random integer
        model_name = args.name if args.name and args.name != "test" else "model"
        # Use current date in seconds as unique identifier
        unique_id = int(time.time())
        unique_directory_name = f"{model_name}_{unique_id}"
        
        # Ensuring the save directory is set and appending the unique directory name
        if args.save_dir is not None:
            args.save_dir = os.path.join(args.save_dir, unique_directory_name)
            # Creating the directory if it doesn't already exist
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir, exist_ok=True)
                logger.info(f"Created new directory for saving checkpoints and models: {args.save_dir}")
            else:
                logger.info(f"Directory already exists. Any saved files will be placed in: {args.save_dir}")
        
    
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()
            
    # initialize wandb without config (it is passed later)
    if global_rank == 0 and args.wandb_project != '':
        wandb_init_kwargs = {
            'project': args.wandb_project
        }
        
        if args.wandb_entity != '':
            wandb_init_kwargs['entity'] = args.wandb_entity
        
        if args.wandb_tag:
            wandb_init_kwargs['tags'] = [args.wandb_tag]
        
        wandb.init(**wandb_init_kwargs)
        
        if args.name != "test":
            wandb.run.name = args.name
            wandb.run.save()
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)


    if args.dataset_name is None:
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
        eval_dataset = None
    # check if folder exists
    elif os.path.exists(args.dataset_name):
        data = datasets.load_dataset(args.dataset_name, split="train", streaming=False)
        eval_dataset = datasets.load_dataset(args.dataset_name, split="validation[:5%]")
        if "text" not in data.column_names:
            data = data.map(lambda x: {"text": x["source"]})
            eval_dataset = eval_dataset.map(lambda x: {"text": x["source"]})
    else:
        data = datasets.load_dataset(args.dataset_name, split="train", streaming=True)
        eval_dataset = datasets.load_dataset(args.dataset_name, split="validation[:5%]")
    
    logger.info(f"Shuffling data with seed {args.seed}")
    data: datasets.Dataset = data.shuffle(seed=args.seed)
    print('Single GPU flag: ', args.single_gpu)
    if not args.single_gpu:
        logger.info(f"Using DDP with world size {world_size}")
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    if args.model_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.max_length)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
        tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)
    
    if args.model_config is not None:
        model_config = AutoConfig.from_pretrained(args.model_config)
        if args.use_hf_model:
            model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
        else:
            model = LlamaForCausalLM(model_config)
    elif args.model_name is not None:
        if args.use_hf_model:
            model = AutoModelForCausalLM.from_pretrained(args.model_name)
        else:
            model = LlamaForCausalLM.from_pretrained(args.model_name)
        model_config = model.config
    else:
        raise ValueError("Either model_config or model_name must be provided")
        
    if args.use_loqt and not args.continue_from:
        logger.info(f"Wrapping model with LoQT")
        model = LoQTModel(
            model, 
            r=args.rank,
            lora_alpha=args.lora_alpha,
            target_modules=["attn", "attention", "mlp"],
            quantize_w=args.quantize_w,
            use_double_quant=args.use_double_quant, 
            device=device,
            proj_type=args.proj_type,
            compute_dtype= torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
            quantize_projection_matrix = args.quantize_projection_matrix,
            compensate_quant_error_iterations = args.compensate_quant_error_iterations,
            use_offloading = args.use_offloading,
            is_single_gpu = args.single_gpu,
            model_config = model_config.to_dict(),
            use_eigenh_for_projection=args.use_eigenh_for_projection,
        )

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        model, global_step, update_step, tokens_seen, tokens_seen_before = load_checkpoint(model, args, logger, device)
        print('Model:', model)
        
        if args.eval_at_continue_from:
            print('Performing Evaluation After Loading Model')
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, tokenizer.pad_token_id, global_rank, world_size, device, args.batch_size, eval_dataset
            )
            # Calculate the perplexity based on the total_loss returned from the evaluation
            perplexity = torch.exp(torch.tensor(total_loss))
            logger.info(f"Eval loss at step {update_step}: {total_loss}, perplexity: {perplexity}")

    update_steps = get_proj_update_steps(args)
    print(f"Projection update steps: {update_steps}")
    
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
        # torch.set_default_dtype(torch.bfloat16)
        print("Model precision: ", model.parameters().__next__().dtype)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
    
    
    # Print parameters and trainable parameters information
    trainable_params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {total_params / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {trainable_params_num / 1_000_000:.2f}M")

    # Setting up optimizers and schedulers
    # Initialize variables
    galore_params = []
    regular_params = []
    layer_wise_flag = False
    
    # GaLore-specific parameter classification and logging
    if 'galore' in args.optimizer.lower():
        regular_params, galore_params = classify_galore_parameters(model)
        logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")

        param_groups = [
            {'params': regular_params},
            {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type, 'update_proj_gap_arr': update_steps if args.proj_gap_progression != 'static' else []}
        ]

    if args.optimizer.lower() in ["adam", "adamw", "sgd", "adafactor", "adam8bit", "galore_adamw", "galore_adafactor", 'galore_adamw8bit']:
        optimizer = create_optimizer(trainable_params if 'galore' not in args.optimizer.lower() else param_groups, args, args.optimizer.lower())
    elif args.optimizer.lower() in ['adamw8bit_per_layer', 'galore_adamw8bit_per_layer']:
        optimizer_dict, scheduler_dict, layer_wise_flag = setup_layerwise_optimizer(model, args, galore_params, update_steps)


    skip_batches = 0 # if continue from set as update_step
    # Load optimizer state if continuing from a checkpoint.
    if args.continue_from is not None:
        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
            
            skip_batches = update_step * args.gradient_accumulation
            del _old_state

        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
            
        #optimizer_state_path = os.path.join(args.continue_from, "optimizer.pt")
        optimizer_state_path = os.path.join(args.continue_from, "optimizer.pt")
        layerwise_optimizer_state_path = os.path.join(args.continue_from, "layerwise_optimizer.pt")
    
        if os.path.exists(optimizer_state_path):
            logger.info(f"Loading optimizer state from {optimizer_state_path}")
            optimizer_state = torch.load(optimizer_state_path, map_location=device)
            optimizer.load_state_dict(optimizer_state['optimizer'])
            del optimizer_state

            logger.info("Optimizer state loaded.")
        elif os.path.exists(layerwise_optimizer_state_path):
            logger.info(f"Loading layerwise optimizer state from {layerwise_optimizer_state_path}")
            optimizer_state = torch.load(layerwise_optimizer_state_path, map_location=device)
            
            for i, (_, opt) in enumerate(optimizer_dict.items()):
                state_key = f"param_{i}"
                if state_key in optimizer_state['layerwise_optimizers']:
                    opt.load_state_dict(optimizer_state['layerwise_optimizers'][state_key])
                else:
                    logger.warning(f"State for key '{state_key}' not found in optimizer_state. Skipping...")
            del optimizer_state
            logger.info("Layerwise optimizer state loaded.")
        else:
            logger.warning(f"No optimizer state found at {optimizer_state_path}")
        
    if not layer_wise_flag:
        total_scheduler_steps = args.num_training_steps
        if args.scheduler_effective_training_steps > 0:
            total_scheduler_steps = args.scheduler_effective_training_steps
        scheduler = training_utils.get_scheduler(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=total_scheduler_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    
    if args.continue_from is not None:
        # Check if the scheduler state is available and load it
        optimizer_state_path = os.path.join(args.continue_from, "optimizer.pt")
        if os.path.exists(optimizer_state_path):
            logger.info(f"Loading optimizer state from {optimizer_state_path}")
            optimizer_state = torch.load(optimizer_state_path, map_location=device)
            if 'scheduler' in optimizer_state:
                scheduler_state = optimizer_state['scheduler']
                scheduler.load_state_dict(scheduler_state)
                logger.info("Scheduler state loaded.")
            else:
                logger.warning("No scheduler state found in the checkpoint.")
            del optimizer_state
                    
    
    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step
    
    if global_rank == 0 and args.log_max_memory:
        # Dynamically get the device used by the model
        device_id = model.device.index if model.device.type == 'cuda' else 0
        logger.info(f'Device for global rank 0 has cuda index {device_id} ')
        device_nvitop = Device(device_id)
        this_process = GpuProcess(os.getpid(), device_nvitop)


    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################
    
    # Placeholder for a temporary scheduler used after merging
    metrics_to_log = {}
    logging_counter_memory_usage_dict = 0
    
    unique_id = int(time.time())
    unique_directory_name = f"loqt_{unique_id}"

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < skip_batches and args.skip_batches_in_continue_from: 
            continue
 
        global_step += 1
        local_step += 1

        should_reset_B = (
            args.use_loqt and 
            update_step in update_steps and
            global_step % args.gradient_accumulation == 0
            and update_step + args.update_proj_gap < args.num_training_steps # do special merge before just before finishing
        )
        
        if should_reset_B:
            
            logger.info("Resetting B matrix")
            actual_model = get_model(model)
            
            dist.barrier()
            start_time_merge = time.time()
            torch.cuda.empty_cache()
            actual_model.merge()
            torch.cuda.empty_cache()

            if not args.single_gpu:
                dist.barrier()
                broadcast_parameters(actual_model, local_rank)

            if not layer_wise_flag:
                optimizer.zero_grad()
            actual_model.set_W_requires_grad(True)
                

        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break
        
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size
        
        if global_rank==0 and args.log_max_memory and update_step > 0 and update_step % args.log_max_memory_steps == 0:
            gpu_metrics = get_gpu_metrics_nvitop(this_process, suffix='_before_backward')
            metrics_to_log.update(gpu_metrics)
            # specific metrics for model, optimizer, and gradients
            if logging_counter_memory_usage_dict < 2: # does not change after first two loggings
                memory_usage_dict = log_memory_usage(get_model(model), optimizer_dict if layer_wise_flag else optimizer, scheduler_dict if layer_wise_flag else scheduler,  args, logger)
                metrics_to_log.update(memory_usage_dict)
                logging_counter_memory_usage_dict +=1

        loss = model(**batch, labels=labels).loss

        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()
        
        if global_rank==0 and args.log_max_memory and update_step > 0 and update_step % args.log_max_memory_steps == 0:
            gpu_metrics = get_gpu_metrics_nvitop(this_process, suffix='_after_backward')
            metrics_to_log.update(gpu_metrics)
            
        if global_rank == 0:
        
            pbar.set_description(f"Update steps, loss: {loss.item():.4f}")                
                
        if global_step % args.gradient_accumulation != 0:
            assert not should_reset_B
            continue

        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0: pbar.update(1)
        
        # The below code is only executed during the update step
        if should_reset_B:
            actual_model = get_model(model)
            actual_model.init_LoRA_with_gradient_projections()
            
            if not layer_wise_flag: 
                optimizer.zero_grad()
            actual_model.set_W_requires_grad(False)
            if not args.single_gpu:
                dist.barrier()
                broadcast_parameters(actual_model, local_rank)
            
            merge_time = time.time() - start_time_merge
            logger.info(f"Merge time: {merge_time:.2f} seconds")
            import gc; gc.collect()
            torch.cuda.empty_cache()

        else:
            if not layer_wise_flag:
                optimizer.step()
                scheduler.step()
                    
                if global_rank==0 and args.log_max_memory and update_step > 0 and update_step % args.log_max_memory_steps == 0:
                    gpu_metrics = get_gpu_metrics_nvitop(this_process, suffix='_after_optimizer_step')
                    metrics_to_log.update(gpu_metrics)
                
                optimizer.zero_grad()
        
        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and args.save_every != 0 and update_step % args.save_every == 0 and global_rank == 0:
            if not args.single_gpu:
                dist.barrier()
                broadcast_parameters(actual_model, local_rank)
            save_checkpoint(
                model,
                optimizer=optimizer_dict if layer_wise_flag else optimizer,
                scheduler=scheduler_dict if layer_wise_flag else scheduler,
                update_step=update_step,
                global_step=global_step,
                run_config=run_config,
                tokens_seen=tokens_seen,
                tokens_seen_before=tokens_seen_before,
                update_time=update_time,
                args=args,
                logger=logger,
                layer_wise_flag=layer_wise_flag,
            )

        # evaluation
        if args.eval_every != 0 and update_step % args.eval_every== 0: # update_step+1 to evaluate just before merging.
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size, eval_dataset
            )
            # Calculate the perplexity based on the total_loss returned from the evaluation
            perplexity = torch.exp(torch.tensor(total_loss))
            if global_rank == 0:
                metrics_to_log.update({
                    "final_eval_loss": total_loss,
                    "final_eval_tokens": evaluated_on_tokens,
                    'final_perplexity': perplexity,
                    })
            logger.info(f"Eval loss at step {update_step}: {total_loss}, perplexity: {perplexity}")

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            metrics_to_log.update({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                })
            wandb.log(metrics_to_log, step=update_step)
            metrics_to_log = {}
            
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")

    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory): 
        if not args.single_gpu:
            dist.barrier()
            broadcast_parameters(actual_model, local_rank)
        save_checkpoint(
                model,
                optimizer=optimizer_dict if layer_wise_flag else optimizer,
                scheduler=scheduler_dict if layer_wise_flag else scheduler,
                update_step=update_step,
                global_step=global_step,
                run_config=run_config,
                tokens_seen=tokens_seen,
                tokens_seen_before=tokens_seen_before,
                update_time=update_time,
                args=args,
                logger=logger,
                layer_wise_flag=layer_wise_flag,
            )


    if args.run_final_eval:
        # Final evaluation
        logger.info("Running final evaluation")
        model.eval()
        if not layer_wise_flag:
            del loss, optimizer, scheduler
        import gc; gc.collect()
        torch.cuda.empty_cache()
    
        total_loss, evaluated_on_tokens = evaluate_model(
            model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
        )
        perplexity = torch.exp(torch.tensor(total_loss))
        if global_rank == 0:
            wandb.log({
                "final_eval_loss": total_loss,
                "final_eval_tokens": evaluated_on_tokens,
                "final_perplexity": perplexity,
                },
                step=update_step,
            )
            logger.info(f"Final eval loss: {total_loss}, and perplexity: {perplexity}")

        logger.info("Script finished successfully")
        print(f"Rank {global_rank} finished successfully")
        


def save_checkpoint(model, optimizer, scheduler, update_step, global_step, run_config, tokens_seen, tokens_seen_before, update_time, args, logger, layer_wise_flag):
                
    latest_checkpoint_directory = f"{args.save_dir}/latest_checkpoint"
    logger.info(f"Overwriting latest model and optimizer at {latest_checkpoint_directory}, update step {update_step}")
    os.makedirs(latest_checkpoint_directory, exist_ok=True)  # Ensures the directory exists, does nothing if already exists

    # Save the actual model
    actual_model = get_model(model)
    if args.save_original_model:
        actual_model.save_pretrained(latest_checkpoint_directory, args.save_original_model)
    else:
        actual_model.save_pretrained(latest_checkpoint_directory) 

    # Save optimizer and scheduler states
    optimizer_checkpoint = {
        "update_step": update_step,
        "global_step": global_step,
        "config": run_config,
        "wandb": wandb.run.dir,
        "dtype": args.dtype,
    }

    if not layer_wise_flag:
        optimizer_checkpoint.update({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        })
        torch.save(optimizer_checkpoint, f"{latest_checkpoint_directory}/optimizer.pt")
    else:
        optimizer_checkpoint.update({"layerwise_optimizers": {f"param_{i}": opt.state_dict() for i, (param, opt) in enumerate(optimizer.items())}})
        torch.save(optimizer_checkpoint, f"{latest_checkpoint_directory}/layerwise_optimizer.pt")



    # Save the training state
    training_state_checkpoint = {
        "global_step": global_step,
        "update_step": update_step,
        "tokens_seen": tokens_seen,
        "tokens_seen_before": tokens_seen_before,
        "update_time": update_time,
    }
    with open(f"{latest_checkpoint_directory}/training_state.json", "w") as f:
        json.dump(training_state_checkpoint, f, indent=4)

    # Save wandb related info in the same directory to keep it updated
    wandb_info = {
        "wandb_id": wandb.run.id,
    }
    with open(f"{latest_checkpoint_directory}/wandb.json", "w") as f:
        json.dump(wandb_info, f, indent=4)

def load_checkpoint(model, args, logger, device):
    logger.info(f"Loading model from {args.continue_from}")
    if args.use_loqt:
        model = LoQTModel.from_pretrained(args.continue_from, device, saved_as_full_model=args.save_original_model)
    else:
        model = load_model_from_checkpoint(args.continue_from, model)

    training_state = os.path.join(args.continue_from, "training_state.json")
    if os.path.exists(training_state):
        with open(training_state) as f:
            state = json.load(f)
        global_step = state["global_step"]
        update_step = state["update_step"]
        tokens_seen = state["tokens_seen"]
        tokens_seen_before = state["tokens_seen_before"]
        logger.info(f"Loaded training state from {training_state}")
        logger.info(f"global_step: {global_step}, update_step: {update_step}, tokens_seen: {tokens_seen}")  
        del state
    else:
        logger.warning(f"No training state found in {training_state}")

    logger.info(f"Model successfully loaded (strict=True policy)")
    return model, global_step, update_step, tokens_seen, tokens_seen_before


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)

