torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --dataset_name mideind/icelandic-common-crawl-corpus-IC3 \
    --use_hf_model \
    --lr 0.0001 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 10000 \
    --batch_size 1 \
    --total_batch_size 512 \
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
    --save_dir checkpoints/llama3_8b_ic3 \
    --proj_gap_progression "exponential" \
    --increment_size 1.15 \
    --name llam3_8b_1.15_100_icelandic \
    --use_offloading True \
    --use_double_quant True \
    --use_loqt True\
    --single_gpu
