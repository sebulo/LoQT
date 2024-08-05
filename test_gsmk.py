import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed

from loqt.LoQT import LoQTModel


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/gpt-neo-1.3B",
        metadata={"help": "Path to the model."},
    )
    ckpt_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to your local output directory"},
    )
    full_precision: bool = field(default=True)
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    
    use_loqt: bool= field(default=True)
    
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be left padded (and possibly truncated)."},
    )

@dataclass
class DataArguments:
    data_name: str = field(default="gsm8k", metadata={"help": "Dataset name."})
    batch_size: int = field(default=16, metadata={"help": "Evaluation batch size."})

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def evaluation(model_args, data_args):
    accelerator = Accelerator()
    device = accelerator.device
    print("Loading model...")
    if model_args.ckpt_dir:
        model_path = model_args.ckpt_dir
    else:
        model_path = model_args.model_name_or_path

    if model_args.use_loqt:
        model = LoQTModel.from_pretrained(model_args.ckpt_dir, device, saved_as_full_model=False)
    elif model_args.full_precision:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            token=model_args.token,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
            ),
        )
        

    print("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        token=model_args.token,
        model_max_length=model_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    if special_tokens_dict:
        print("Resizing tokenizer and embeddings...")
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

    model = model.to(device)

    ######################
    #      dataset       #
    ######################
    print("Downloading dataset...")
    dataset = load_dataset(data_args.data_name, "main")
    test_set = dataset['test']

    print("Formatting inputs...")
    question = [f"{example['question']}{QUESTION_PROMPT}" for example in test_set]
    answer = []

    # get numerical answer
    for example in test_set['answer']:
        ans = example.split('####')[-1]
        ans = ans.replace(',', '')  # handle numbers like 2,000
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    print("Tokenizing inputs...")
    eval_step = math.ceil(len(question) / data_args.batch_size)
    print(f"Total examples: {len(question)} | Eval batch size: {data_args.batch_size} | Eval steps: {eval_step}")
    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i * data_args.batch_size: (i + 1) * data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i * data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch)

    print("Starting evaluation...")
    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }
    ans_pred_list = []
    set_seed(42)
    for step, batch in enumerate(question_data):
        with torch.no_grad():
            gen_kwargs["input_ids"] = batch["input_ids"].to(device)
            gen_kwargs["attention_mask"] = batch["attention_mask"].to(device)
            generated_tokens = model.generate(**gen_kwargs)

        pred_tokens = generated_tokens[:, batch['input_len']:]
        decoded_pred = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

        # Extract the numbers in sentences
        print(f"Decoded predictions for step {step}: {decoded_pred}")
        ans_pred_list += [extract_answer_number(sentence_pred) for sentence_pred in decoded_pred]

    print("Prediction:", ans_pred_list)
    print("Ground truth:", answer)

    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"Model: {model_args.model_name_or_path} | GSM8K test accuracy: {100 * accuracy:.2f}% | "
          f"Full precision: {model_args.full_precision}")

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def compute_accuracy(pred: list, gold: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1

    return acc / len(pred)

if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    print("Parsed arguments:", model_args, data_args)
    if model_args.ckpt_dir is not None:
        print(f"Using checkpoint directory: {model_args.ckpt_dir}")
    evaluation(model_args, data_args)