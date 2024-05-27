#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --output=output_files/job.%j.out      # Name of output file (%j expands to jobId)
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:titanx:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=72:00:00


echo "Running on $(hostname):"
nvidia-smi

eval "$(conda shell.bash hook)"
conda activate galora


torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name mideind/icelandic-common-crawl-corpus-IC3 \
    --use_hf_model \
    --lr 0.0001 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 4 \
    --batch_size 1 \
    --total_batch_size 2 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --dtype bfloat16 \
    --eval_every 2000 \
    --optimizer adam8bit \
    --lora_alpha 0.5 \
    --bnb_4bit_quant_type nf4 \
    --quantize_w '4bit' \
    --quantize_projection_matrix '4bit' \
    --seed 42 \
    --save_every 1000 \
    --save_dir checkpoints/llama2_7b_ic3 \
    --proj_gap_progression "exponential" \
    --increment_size 1.15 \
    --name llam2_7b_1.15_100_icelandic \
    --use_offloading True \
    --use_double_quant True \
    --use_loqt True \
    --single_gpu 
