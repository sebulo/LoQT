# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from loqt.utils import get_proj_update_steps, filter_linear_target_modules
from peft import LoraConfig, get_peft_model



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
        choices=list(task_to_keys.keys()),
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
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
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
    
    # support enable_galore
    parser.add_argument("--enable_galore", action="store_true", help="Whether or not to use low rank optimizer.")
    # update_proj_gap
    parser.add_argument("--update_proj_gap", type=int, default=50)
    # galore_scale
    parser.add_argument("--galore_scale", type=float, default=1.0)
    # proj_type
    parser.add_argument("--proj_type", type=str, default="std")
    # lora_all_modules
    parser.add_argument("--lora_all_modules", default=True, type=lambda x: x.lower() == "true")
    # eval_llama
    parser.add_argument("--eval_llama", action="store_true", help="Whether or not to evaluate llama model.")
    # low_rank_method
    parser.add_argument("--low_rank_method", type=str, default=None, help="low rank method for wandb sweep")
    
    # LoQT args
    parser.add_argument("--use_loqt", default=False, type=lambda x: x.lower() == "true", help="Enable LoQT")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha value")
    parser.add_argument("--only_train_lora", default=True, type=lambda x: x.lower() == "true", help="Train only LoRA layers")
    parser.add_argument("--use_eigenh_for_projection", default=False, type=lambda x: x.lower() == "true", help="Use EigenH for projection, if false use SVD")
    parser.add_argument('--use_offloading', default=False, type=lambda x: x.lower() == "true", help="Enable offloading")
    
    # Quantization Parameters
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

    # Regular LoRA Parameter
    parser.add_argument("--init_lora_weights", type=str, default="gaussian", help="Initialization strategy for LoRA weights")
    parser.add_argument("--use_regular_lora", action="store_true", help="Use regular LoRA")

    
    args = parser.parse_args()
    
    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("loqt_benchmark_GLUE_final", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
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
    
    
    
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
            
    
    
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if not args.eval_llama:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            trust_remote_code=args.trust_remote_code,
        )
    elif 'deberta' in args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes, 
            )

    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        setattr(config, 'num_labels', num_labels)
        setattr(config, 'finetuning_task', args.task_name)
        tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
        tokenizer.padding_side = "left"
        model = LlamaForSequenceClassification(
            config
        )
        
    ## load pretrained model
    if args.load_pretrained_model:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.load_pretrained_model}")
        checkpoint_path = os.path.join(args.load_pretrained_model, "pytorch_model.bin")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        for key in checkpoint.keys():
            if key not in model.state_dict().keys():
                print(f"key {key} not in model state dict")
        
        for key in model.state_dict().keys():
            if key not in checkpoint.keys():
                print(f"key {key} not in checkpoint")
        model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Model successfully loaded (strict=False policy)")
        logger.info("*" * 40)
        
    # get dtypes of model
    logger.info(f"Model dtype: {model.dtype}")
    dtype = model.dtype
    
    # project modules
    if 'roberta' in args.model_name_or_path:
        if args.lora_all_modules:
            target_modules_list = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj", "k_proj", "o_proj"]
        else:
            target_modules_list = ["q_proj", "v_proj"]
    if 'deberta' in args.model_name_or_path:
        if args.lora_all_modules:
            target_modules_list = ['query', 'key', 'value',
                        'q_proj', 'k_proj', 'v_proj',
                        'query_proj', 'key_proj', 'value_proj',
                        'out_proj', 'dense', 'attention', 'fc1', 'fc2']
        else:
            target_modules_list = ['query_proj', 'key_proj', 'value_proj']
        
        
    # After loading the pre-trained model
    if args.use_loqt:
        for param in model.parameters():
            param.requires_grad = False
        logger.info(f"Wrapping model with LoQT")
        model = LoQTModel(
            model, 
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules_list, #TODO in pretraining we use ["attn", "attention", "mlp"],
            quantize_w=args.quantize_w,
            use_double_quant=args.use_double_quant, 
            device=device,
            proj_type=args.proj_type,
            compute_dtype=dtype, #compute_dtype= torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
            quantize_projection_matrix = args.quantize_projection_matrix,
            compensate_quant_error_iterations = args.compensate_quant_error_iterations,
            is_single_gpu= args.single_gpu,
            only_train_lora=args.only_train_lora,
        )  
    elif args.use_regular_lora:
        target_modules_list = filter_linear_target_modules(model, target_modules_list)
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            init_lora_weights=args.init_lora_weights,
            target_modules=target_modules_list,
            bias="none",
            task_type="CAUSAL_LM",  # or "SEQUENCE_CLASSIFICATION" if that's the correct task type
        )
        model = get_peft_model(model, lora_config)
        logger.info("Model wrapped with regular LoRA")
        show_model_stats_deberta(model, mark_only_lora_as_trainable=True)
        
    
    if 'deberta' in args.model_name_or_path:  
        if not args.use_loqt:
            # no wrapped model
            model.classifier.weight.requires_grad = True
            model.classifier.bias.requires_grad = True
        else:
            model.wrapped_model.classifier.weight.requires_grad = True
            model.wrapped_model.classifier.bias.requires_grad = True
    
    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Get the configuration from the correct model reference
    config = get_model_config(model)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if args.task_name is not None and not is_regression:
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        set_model_labels(model, label_to_id, {id: label for label, id in label_to_id.items()})
    elif args.task_name is not None and not is_regression:
        new_label_to_id = {l: i for i, l in enumerate(label_list)}
        set_model_labels(model, new_label_to_id, {id: label for label, id in new_label_to_id.items()})

    padding = "max_length" if args.pad_to_max_length else False
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    if args.use_loqt or args.use_regular_lora:
        params1 = [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)]
        params2 = [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)]
    else:
        params1 = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params2 = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {
            "params": params1,
            "weight_decay": args.weight_decay,
        },
        {
            "params": params2,
            "weight_decay": 0.0,
        },
    ]
    print('init lr: ', args.learning_rate)
    
    
    # math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        args.num_training_steps = args.max_train_steps #used by get_projection_update_steps
        overrode_max_train_steps = True
    
    update_steps = get_proj_update_steps(args)
    print('update_steps: ', update_steps)
    
    if not args.enable_galore:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        from torch import nn
        galore_params = []
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)

        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [{'params': regular_params}, 
                        {'params': galore_params, 'rank': args.lora_r, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type,  'update_proj_gap_arr': update_steps if args.proj_gap_progression != 'static' else []}]
        optimizer = GaLoreAdamW(param_groups, lr=args.learning_rate)
    

    # check number of trainable params in optimizer
    num_trainable_params = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            num_trainable_params += p.numel()
    print('num_trainable_params in optimizer: ', num_trainable_params)

    # Scheduler 
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("loqt_benchmark_GLUE_final", experiment_config)

    # Get the metric function
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")

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
    best_eval_loss = float('inf')  # Initialize with infinity, assuming lower is better
    best_eval_metric1 = float('-inf')  # Initialize to the worst possible value for maximization
    best_eval_metric2 = float('-inf')  # Initialize only if needed
    current_metric1 = 0
    # Assuming args.task_name contains the current task name
    metrics_names = task_metrics[args.task_name]  # Corrected to access dictionary values


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        
        logger.info(f"***** Epoch {epoch} *****")
        
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            # completed_steps is update_step and global_step is step
            should_reset_B = (
                args.use_loqt and 
                completed_steps in update_steps and
                completed_steps % args.update_proj_gap == 0 and 
                step % args.gradient_accumulation_steps == 0
            )
            
            if should_reset_B:
                logger.info(f"Resetting B matrix at step {completed_steps}")
                model.merge()
                optimizer.zero_grad()
                model.set_W_requires_grad(True)
                
                model.set_LoRA_requires_grad(True)
                model.disable_lora(False)
                model.lora_zero_init()
                
            outputs = model(**batch)
            #logger.info(f"outputs: {outputs}")
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if should_reset_B:
                model.reinitialize_LoRA_AB_after_merge()
                optimizer.zero_grad()
                model.set_W_requires_grad(False)
                model.set_LoRA_requires_grad(True)
                model.disable_lora(False)
                print('num_trainable_params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
                if 'deberta' in args.model_name_or_path:  
                    if not args.use_loqt:
                        model.classifier.weight.requires_grad = True
                        model.classifier.bias.requires_grad = True
                    else:
                        model.wrapped_model.classifier.weight.requires_grad = True
                        model.wrapped_model.classifier.bias.requires_grad = True
            elif step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
            
            # log loss
            logger.info(f"epoch {epoch}, step {step}: {loss.item()}")


        model.eval()
        samples_seen = 0
        total_eval_loss = 0
        eval_steps = 0
        
        
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                
            loss = outputs.loss.item()
            total_eval_loss += loss
            eval_steps += 1
            
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            
        

        eval_metric = metric.compute()
        average_eval_loss = total_eval_loss / eval_steps
        logger.info(f"Epoch {epoch}: {eval_metric}, Eval Loss: {average_eval_loss}")

        # Initialize current metrics
        current_metric1 = eval_metric.get(metrics_names[0], float('-inf'))
        current_metric2 = None

        # Check if the current model is the best based on evaluation loss
        if average_eval_loss < best_eval_loss:
            best_eval_loss = average_eval_loss
            logger.info(f"New best model found at epoch {epoch} with eval loss {average_eval_loss}")

        # Log and track best metrics as per task requirements
        if metrics_names[0] in eval_metric:
            current_metric1 = eval_metric[metrics_names[0]]
            if current_metric1 > best_eval_metric1:
                best_eval_metric1 = current_metric1
                logger.info(f"New best model found at epoch {epoch} with {metrics_names[0]} {current_metric1}")
        else:
            logger.warning(f"{metrics_names[0]} key not found in eval metrics.")

        # If there is a second metric to track
        if len(metrics_names) > 1 and metrics_names[1] in eval_metric:
            current_metric2 = eval_metric[metrics_names[1]]
            if current_metric2 > best_eval_metric2:
                best_eval_metric2 = current_metric2
                logger.info(f"New best model found at epoch {epoch} with {metrics_names[1]} {current_metric2}")

        # Log metrics using wandb or another tracker if with_tracking is enabled
        if args.with_tracking:
            log_data = {
                "train_loss": total_loss / len(train_dataloader),  # Assuming total_loss is tracked during training
                "eval_loss": average_eval_loss,
                "best_eval_loss": best_eval_loss,
                metrics_names[0]: current_metric1,
                "best_" + metrics_names[0]: best_eval_metric1,
                "epoch": epoch
            }

            # Include second metric if it exists and is defined
            if len(metrics_names) > 1 and current_metric2 is not None:
                log_data[metrics_names[1]] = current_metric2
                log_data["best_" + metrics_names[1]] = best_eval_metric2

            accelerator.log(log_data, step=completed_steps)

            
        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            # Unwrap the model for saving if not using LoQT; use custom save if LoQT is used
            if not args.use_loqt:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
            else:
                # Use your custom save function for LoQT
                model.save_pretrained(args.output_dir)
                
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)


    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if not args.use_loqt:
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
        else:
            # Use your custom save function for LoQT
            model.save_pretrained(args.output_dir)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)

# Helper function to get model configuration
def get_model_config(model):
    if hasattr(model, 'wrapped_model'):
        return model.wrapped_model.config
    else:
        return model.config

# Helper function to set model configuration
def set_model_labels(model, label2id, id2label):
    if hasattr(model, 'wrapped_model'):
        model.wrapped_model.config.label2id = label2id
        model.wrapped_model.config.id2label = id2label
    else:
        model.config.label2id = label2id
        model.config.id2label = id2label

# from https://github.com/yxli2123/LoftQ/blob/main/glue/utils.py
def show_model_stats_deberta(model,mark_only_lora_as_trainable=True):
    total = 0
    lr_adapter = 0
    if mark_only_lora_as_trainable:
        for n, m in model.deberta.named_parameters():
            if 'lora' in n or 'left' in n or 'right' in n:
                m.requires_grad = True
                lr_adapter += m.numel()
            else:
                if "quant" in n or "word_embeddings.weight" in n:
                    print(n, m)
                m.requires_grad = False
            print(n, m.shape, m.requires_grad)
            total += m.numel()
    else:
        for n, m in model.deberta.named_parameters():
            if "quant" in n or "word_embeddings.weight" in n:
                print(n, m)
            if m.requires_grad:
                lr_adapter += m.numel()
                print(lr_adapter)
            total += m.numel()
    print(f"Total trainable parameters {lr_adapter}")
    print(f"We finetune about {lr_adapter / total} ratio of percentages")

        
if __name__ == "__main__":
    main()