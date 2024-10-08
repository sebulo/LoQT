torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    --model_name /store/swissai/a06/models/Meta-Llama-3-8B \
    --dataset_name /iopsstor/scratch/cscs/vsnbjarn/raw_pretraining_data \
    --use_hf_model \
    --lr 0.0001 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 10000 \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 1500 \
    --warmup_steps 500 \
    --dtype bfloat16 \
    --eval_every 2000 \
    --optimizer adam8bit \
    --lora_alpha 0.5 \
    --bnb_4bit_quant_type nf4 \
    --quantize_w '4bit' \
    --quantize_projection_matrix '4bit' \
    --seed 42 \
    --save_every 1000 \
    --save_dir checkpoints/llama3_8b_ic3 \
    --proj_gap_progression "exponential" \
    --increment_size 1.15 \
    --name llam3_8b_1.15_100_icelandic \
    --use_double_quant True \
    --use_loqt True

#--single_gpu