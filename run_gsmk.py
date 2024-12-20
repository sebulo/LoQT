""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split

import datasets
# import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from loqt.utils import get_proj_update_steps, filter_linear_target_modules
# from peft import LoraConfig, get_peft_model, PeftModel, LoraConfig, TaskType, get_peft_model

# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset


import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForSequenceClassification
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from optimizers import GaLoreAdamW

from loqt.LoQT import LoQTModel

# for testing
from eval_gsmk import evaluation, ModelArguments, DataArguments

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.38.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_metrics = {
    "cola": ("matthews_correlation",),
    "mnli": ("accuracy",),
    "mrpc": ("accuracy", "f1"),
    "qnli": ("accuracy",),
    "qqp": ("accuracy", "f1"),
    "rte": ("accuracy",),
    "sst2": ("accuracy",),
    "stsb": ("pearson", "spearmanr"),
    "wnli": ("accuracy",),
}

def set_seed_torch(seed: int):

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    
    

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--load_pretrained_model", type=str, default=None)

    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli", "gsmk"],
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps_procentage", type=float, default=0.0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    
    # Additional arguments for LoQT, LoRA, etc.
    parser.add_argument("--enable_galore", action="store_true", help="Whether or not to use low rank optimizer.")
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    parser.add_argument("--lora_all_modules", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--eval_llama", action="store_true", help="Whether or not to evaluate llama model.")
    parser.add_argument("--low_rank_method", type=str, default=None, help="low rank method for wandb sweep")
    parser.add_argument("--use_loqt", default=False, type=lambda x: x.lower() == "true", help="Enable LoQT")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha value")
    parser.add_argument("--only_train_lora", default=False, type=lambda x: x.lower() == "true", help="Train only LoRA layers")
    parser.add_argument("--use_eigenh_for_projection", default=False, type=lambda x: x.lower() == "true", help="Use EigenH for projection, if false use SVD")
    parser.add_argument('--use_offloading', default=False, type=lambda x: x.lower() == "true", help="Enable offloading")
    parser.add_argument("--quantize_w", type=str, default=None, choices=["1bit", "4bit", "8bit"], help="Quantization level for weights")
    parser.add_argument("--use_double_quant", default=False, type=lambda x: x.lower() == "true", help="Enable double quantization")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"], help="4-bit quantization type")
    parser.add_argument('--quantize_projection_matrix', default=None, type=str, help='4bit for 4bit quantization of projection matrix')
    parser.add_argument('--compensate_quant_error_iterations', type=int, default=0, help='Number of iterations to run the joint optimization of LoRA/quant')
    parser.add_argument('--proj_gap_progression', type=str, default="static", choices=["static", "linear", "exponential"], help="Projection gap progression strategy")
    parser.add_argument('--increment_size', type=float, default=1.2, help="Factor for increasing warmup steps either linear steps or exponential factor")
    parser.add_argument('--max_proj_gap', type=float, default=0, help="Maximum projection gap")
    # General Training Parameters
    parser.add_argument("--single_gpu", default=False, action="store_true", help="flag for LoQT to use distributed or not")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32", help="Data type for training")
    parser.add_argument("--log_loss_every", type=int, default = 50)
    
    parser.add_argument("--save_original_model", default=True, action="store_true", help="flag for LoQT to also save full model and not just model + adapters")
    parser.add_argument("--experiment_name", type=str, default="" )
    parser.add_argument("--train_all_params", default=True)
    parser.add_argument('--eval_subset_dataset', type=float, default=1.0, help="subset to test on")
    parser.add_argument("--run_eval_every_epoch", type=int, default = 1)
    parser.add_argument("--val_split_percentage", type=float, default=0.0, help="Percentage of the training dataset to use for validation")
    
    
    return parser.parse_args()




IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


def set_seed_torch(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model_to_resize = model.wrapped_model if hasattr(model, "wrapped_model") else model
    model_to_resize.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model_to_resize.get_input_embeddings().weight.data
        output_embeddings = model_to_resize.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: AutoTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    # sources are questions, and targets are answers
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)    


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,  # Ensure truncation is applied here
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

    
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: AutoTokenizer):
        super(SupervisedDataset, self).__init__()

        logger.warning("Formatting inputs...")
        sources = [f"{example['question']}{QUESTION_PROMPT}" for example in raw_data]
        targets = [f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT) for example in raw_data]

        logger.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {"input_ids": [], "labels": []}

        for instance in instances:
            batch["input_ids"].append(instance["input_ids"])
            batch["labels"].append(instance["labels"])

        # Padding the input_ids and labels
        batch["input_ids"] = self.tokenizer.pad(
            {"input_ids": batch["input_ids"]},
            padding='longest',
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )["input_ids"]

        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            batch["labels"], batch_first=True, padding_value=IGNORE_INDEX
        )

        # Creating attention masks
        batch["attention_mask"] = batch["input_ids"].ne(self.tokenizer.pad_token_id)

        return batch

def make_supervised_data_module(tokenizer: AutoTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logger.warning("Downloading Data")
    train_set = load_dataset("gsm8k", "main", split="train")

    # Split into train and validation sets
    train_indices, val_indices = train_test_split(
        range(len(train_set)), 
        test_size=data_args.val_split_percentage, 
        random_state=42
    )
    
    # Use Subset to create new train and validation sets
    train_subset = torch.utils.data.Subset(train_set, train_indices)
    val_subset = torch.utils.data.Subset(train_set, val_indices)
    
    train_dataset = SupervisedDataset(raw_data=train_subset, tokenizer=tokenizer)
    val_dataset = SupervisedDataset(raw_data=val_subset, tokenizer=tokenizer)
    
    print(f"Train dataset size: {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)}")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)

def make_supervised_data_module(tokenizer: AutoTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logger.warning("Downloading Data")
    train_set = load_dataset("gsm8k", "main", split="train")

    # Initialize the validation dataset
    val_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # Split into train and validation sets only if val_split_percentage is greater than 0
    if data_args.val_split_percentage > 0:
        train_indices, val_indices = train_test_split(
            range(len(train_set)), 
            test_size=data_args.val_split_percentage, 
            random_state=42
        )
        
        # Use Subset to create new train and validation sets
        train_subset = torch.utils.data.Subset(train_set, train_indices)
        val_subset = torch.utils.data.Subset(train_set, val_indices)

        print(f"Train dataset size: {len(train_subset)}")
        print(f"Validation dataset size: {len(val_subset)}")
    else:
        # If no validation split, use the entire train set
        train_subset = train_set
        print(f"Train dataset size: {len(train_set)}")

    # Create the train dataset and the validation dataset (if it exists)
    train_dataset = SupervisedDataset(raw_data=train_subset, tokenizer=tokenizer)
    if data_args.val_split_percentage > 0:
        val_dataset = SupervisedDataset(raw_data=val_subset, tokenizer=tokenizer)
        return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)
    else: 
        return dict(train_dataset=train_dataset,data_collator=data_collator)

    


def main():
    args = parse_args()
    
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("loqt_benchmark_gsmk", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        model_name_trimmed = args.model_name_or_path.replace("/", "_")
        experiment_name = f"{model_name_trimmed}_GSMK"
    
    unique_output_dir = os.path.join(args.output_dir, f"{experiment_name}_epochs{args.num_train_epochs}_seed{args.seed}{current_time}")
    args.output_dir = unique_output_dir
    output_dir = args.output_dir
    print('output_dir: ', output_dir)
    
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(experiment_name, experiment_config)
    print("Accelerator State:", accelerator.state)
    print("Device Setup by Accelerator:", accelerator.device)

    # Accessing the device information
    device = accelerator.device
    print(f"Using device: {device}")
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        set_seed_torch(args.seed)
        

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
        
    #task_type = TaskType.CAUSAL_LM TODO should be used?
    if any(name in args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "gpt2", "gpt-neo"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif any(name in args.model_name_or_path.lower() for name in ["phi"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    else:
        raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {args.model_name_or_path}.")
    
    if args.use_loqt:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=args.hub_token,
        )
        if args.train_all_params:
            # TODO is this what we want ?
            for param in model.parameters():
                param.requires_grad = False
        model = LoQTModel(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            quantize_w=args.quantize_w,
            use_double_quant=args.use_double_quant,
            device=device,
            proj_type=args.proj_type,
            compute_dtype=torch.bfloat16,
            quantize_projection_matrix=args.quantize_projection_matrix,
            compensate_quant_error_iterations=args.compensate_quant_error_iterations,
            is_single_gpu=args.single_gpu,
            only_train_lora=args.only_train_lora,
            use_offloading=args.use_offloading,
            grad_accumulation_steps=args.gradient_accumulation_steps,
        )

    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=args.hub_token,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=args.use_double_quant,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            ),
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        token=args.hub_token,
        cache_dir=args.output_dir,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "[PAD]"
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "</s>"
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = "<s>"
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = "<unk>"
        

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # Data Preparation
    
    data_module = make_supervised_data_module(tokenizer, args)
    train_dataset = data_module['train_dataset']
    if args.val_split_percentage > 0:
        eval_dataset = data_module['eval_dataset']
    data_collator = data_module['data_collator']
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=args.per_device_train_batch_size
    )
    if args.val_split_percentage > 0:
        validation_dataloader = DataLoader(
            eval_dataset, 
            collate_fn=data_collator, 
            batch_size=args.per_device_eval_batch_size
        )
    
    # # Optimizer 
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Determine the number of update steps per epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_training_steps = args.max_train_steps  # Used by get_projection_update_steps

    update_steps = get_proj_update_steps(args)
    print('update_steps: ', update_steps)

    num_warmup_steps = args.num_warmup_steps_procentage * args.max_train_steps
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.val_split_percentage > 0:
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
            )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, lr_scheduler
            )
    ##### TRAINABLE PARAMS #####
    # Check number of trainable params in optimizer
    num_trainable_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    print('num_trainable_params in optimizer: ', num_trainable_params)

    # Recalculate total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Log the calculations for debugging purposes
    logger.info(f"Number of update steps per epoch: {num_update_steps_per_epoch}")
    logger.info(f"Max training steps: {args.max_train_steps}")
    logger.info(f"Number of training epochs: {args.num_train_epochs}")

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_acc = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    
    # Initialize best_val_loss and best_model_path
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, "best_model")
    final_best_model_accuracy = None

    # Your training loop
    for epoch in range(starting_epoch, args.num_train_epochs):
        logger.info(f"***** Epoch {epoch} *****")

        model.train()
        if args.with_tracking:
            total_loss = 0
        
        active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):

            outputs = model(**batch)
            loss = outputs.loss
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            
            if step % args.gradient_accumulation_steps != 0:
                continue

            elif (step+1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

            if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps}"
                if args.output_dir:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
            
            if step % args.log_loss_every == 0:
                logger.info(f"epoch {epoch}, step {step}: {loss.item()}")
                if args.with_tracking:
                    accelerator.log({"train_loss": loss.item(), "step": completed_steps})
                    
        # Save the best model (overwriting the previous best)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if not args.use_loqt:
            unwrapped_model.save_pretrained(
                best_model_path, 
                is_main_process=accelerator.is_main_process, 
                save_function=accelerator.save
            )
        else:
            model.save_pretrained(
                best_model_path, 
                save_original_model=args.save_original_model, 
                only_save_original_model=True
            )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(best_model_path)
        #logger.info(f"New best model saved at epoch {epoch} with validation loss: {best_val_loss}")

        if epoch > 0 and ((epoch % args.run_eval_every_epoch == 0) or (epoch == args.num_train_epochs - 1)):
            print('running evaluation')
            accuracy = run_evaluation(args.output_dir, model, tokenizer, device, args)
            if accuracy > best_acc:
                best_acc = accuracy
            accelerator.log({"eval accuracy": accuracy, "step": completed_steps, "epoch": epoch, "best eval accuracy": best_acc})
            logger.info(f"epoch {epoch}, acc GSM8K test accuracy: {100 * accuracy:.2f}%")
            print(f"epoch {epoch}, acc GSM8K test accuracy: {100 * accuracy:.2f}%")
            # did eval on last epoc
            if epoch == args.num_train_epochs - 1 and args.val_split_percentage == 0:
                final_best_model_accuracy = accuracy
                logger.info(f"Final best model accuracy: {final_best_model_accuracy}")
                accelerator.log({"final_best_model_accuracy": final_best_model_accuracy})
                # After training loop
                logger.info(f"Best model was saved at: {best_model_path}")


        if args.with_tracking:
            log_data = {
                "train_loss": total_loss / len(train_dataloader),
                "epoch": epoch,
            }
            accelerator.log(log_data, step=completed_steps)
            logger.info(f"epoch {epoch}: {total_loss / len(train_dataloader)}")

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            

    if args.val_split_percentage >0:
        best_model = LoQTModel.from_pretrained(best_model_path, device=device, saved_as_full_model=False)
    else:
        best_model = model
    if final_best_model_accuracy is None:
        final_best_model_accuracy = run_evaluation(args.output_dir, best_model, tokenizer, device, args, is_loqt=False)
    logger.info(f"Final best model accuracy: {final_best_model_accuracy}")
    accelerator.log({"final_best_model_accuracy": final_best_model_accuracy})
    # After training loop
    logger.info(f"Best model was saved at: {best_model_path}")

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if not args.use_loqt:
            unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
        else:
            model.save_pretrained(args.output_dir, save_original_model=args.save_original_model, only_save_original_model=True)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
                
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

def run_evaluation(model_dir, model,  tokenizer, device, args, is_loqt = True ):
    # Find the latest checkpoint directory
    print(f"Using latest checkpoint directory for evaluation: {model_dir}")
    
    # get full model
    if is_loqt:
        org_model = model.return_original_model()
    else:
        org_model = model
    
    # Move the model off the device
    model.to('cpu')
    
    tokenizer.save_pretrained(args.output_dir)
    # Create model_args and data_args for evaluation
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        ckpt_dir=model_dir,
        full_precision=True,
        token='x',
        model_max_length=args.max_length,
        use_loqt=args.use_loqt
    )
    
    data_args = DataArguments(
        data_name="gsm8k",
        batch_size=args.per_device_eval_batch_size  # Adjust batch size as needed
    )
    
    # Call the evaluation function
    accuracy = evaluation(model_args, data_args, model =org_model, print_decoding = False, subset_dataset = args.eval_subset_dataset)
    
    model.to(device)
    return accuracy
    
# Helper function to get model configuration
def get_model_config(model):
    if hasattr(model, 'wrapped_model'):
        return model.wrapped_model.config
    else:
        return model.config
        
if __name__ == "__main__":
    main()