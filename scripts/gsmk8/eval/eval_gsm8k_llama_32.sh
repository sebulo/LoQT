#!/bin/bash

#BSUB -q p1                          # Specify queue
#BSUB -J simple_gpu_job              # Set the job name
#BSUB -n 16                          # Request number of cores (default: 1)
#BSUB -R "span[hosts=1]"             # Specify cores must be on the same host
#BSUB -R "rusage[mem=32GB]"           # Specify 4GB of memory per core/slot
#BSUB -W 72:00                       # Set walltime limit: hh:mm#BSUB -B
#BSUB -o output_files/job.%J.out     # Specify the output file. %J is the job-id
#BSUB -e output_files/job.%J.err     # Specify the error file. %J is the job-id

# Requesting GPU resources
#BSUB -gpu "num=1:j_exclusive=yes"   # Request 1 GPU, with exclusive access


echo "Running on $(hostname):"
nvidia-smi

eval "$(conda shell.bash hook)"
conda activate loqt


python test_gsmk.py\
  --model_name_or_path checkpoints/llama7b \
  --ckpt_dir checkpoints/meta-llama_Llama-2-7b-hf_GSMK_epochs6_seed1120240805_224459 \
  --use_loqt True \
  --batch_size 32 

  #--ckpt_dir checkpoints/meta-llama_Llama-2-7b-hf_GSMK_epochs3_seed1120240805_203109 \ #

  #--ckpt_dir checkpoints/meta-llama_Llama-2-7b-hf_GSMK_epochs6_seed1120240805_172950 \
  #--ckpt_dir checkpoints/meta-llama_Llama-2-7b-hf_GSMK_epochs6_seed1120240805_172553 \
  #--ckpt_dir checkpoints/meta-llama_Llama-2-7b-hf_GSMK_epochs6_seed1120240805_173133 \

  #--ckpt_dir checkpoints/meta-llama_Llama-2-7b-hf_GSMK_epochs5_seed987620240805_141153 \