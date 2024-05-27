# LoQT

This repository contains the pre-release version of the code accompanying the paper "LoQT: Low Rank Adapters for Quantized Training". Note that this is an early version of the codebase.

LoQT is a method for training quantized models with low-rank adapters, aimed at reducing the number of parameters in large language models. This method is implemented in PyTorch and enables efficient quantized pre-training and fine-tuning of models, achieving results close to full-rank, non-quantized models. 

LoQT allows for the pre-training of a 13B LLM on a 24GB GPU without model parallelism, checkpointing, or offloading strategies during training.


## Table of Contents

- [Setup](#setup)
- [Usage Examples](#usage-examples)
- [Memory Usage](#memory-usage)
- [Benchmarks](#benchmarks)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Setup

To install the dependencies, run the following command:

```sh
conda env create -f environment.yml
```
This will create a conda environment with all the necessary packages. Make sure to activate the environment:


```
conda activate loqt
```

## Usage Examples



Pre-training

To run a sample script for pre-training a 350m Llama 2 model, use the following command:

```sh
bash scripts/benchmark_pretraining/350m_loqt8bit.sh
```

Fine-tuning

To run a sample script for fine-tuning, use the following command:

```sh
bash scripts/benchmark_finetuning/finetune_cola.sh
```

To run a specific training configuration, you can use the following command:
      
```sh
torchrun --standalone --nproc_per_node 1 --nnodes 1 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --seed 42 \
    --lr 0.01 \
    --rank 256 \
    --lora_alpha 0.5 \
    --update_proj_gap 100 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --eval_every 2000 \
    --save_every 10000 \
    --dtype bfloat16 \
    --use_loqt True \
    --quantize_w 4bit \
    --quantize_projection_matrix '4bit' \
    --compensate_quant_error_iterations 5 \
    --proj_gap_progression "exponential" \
    --increment_size 1.2 \
    --single_gpu \
    --optimizer adam8bit \
    --name 350m_LoQT_8bit

```

Continued pre-training example is available in the `scripts/continued_pretraining` folder.

More details can be found in torchrun_main.py for pretraining and run_glue.py for fine-tuning.

## Memory Usage
To compare memory usage across different model configuraions, you can run the scripts in the folder `memory_profiling`. This script `all.sh` logs the memory usage of the model with and without quantization for LoQT, GaLore, and the regular Adam optimizer. The memory profiling is done for both 16-bit and 8-bit optimizers. Additionally, for GaLore and LoQT, per-layer gradient updates are also run.
`13b_rank1024_loqt.sh` logs the memory usage of the 13B model with a rank of 1024 for LoQT.

### Running the Script

Execute the following command to run the memory profiling script:

```sh
bash scripts/memory_profiling.sh
```
### Script Details
The script contains a line that declares an array of models and their corresponding ranks:

```sh
declare -a models_and_ranks=("MODEL_CONFIG:MODEL_RANK")
```
* MODEL_CONFIG: The name of the model configuration file.
* MODEL_RANK: The rank of the model.

For example, to test the memory usage for the llama_350m model with a rank of 256, you would modify the line as follows:
```sh
declare -a models_and_ranks=("llama_350m:256")
```
If you want to test the memory usage for the llama_7b model with a rank of 1024, you would use:
```sh
declare -a models_and_ranks=("llama_7b:1024")
```

## Benchmarks
To run the benchmark pre-training scripts, navigate to the benchmark_pretraining folder. The scripts are named after the model size.

For fine-tuning benchmarks using DeBERTa-v3 on GLUE tasks, navigate to the finetuning_deberta folder. The scripts are named after the task.

## Acknowledgements 
Parts of the code are based on the repository by Jiawei Zhao et al.: https://github.com/jiaweizhao/GaLore


## Citation
If you use this codebase in your work, please cite our paper:
```bibtex
@misc{loeschcke2024loqt,
      title={LoQT: Low Rank Adapters for Quantized Training},
      author={Sebastian Loeschcke and Mads Toftrup and Michael Kastoryano and Serge Belongie and Vésteinn Snæbjarnarson},
      year={2024},
      eprint={INSERT},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
