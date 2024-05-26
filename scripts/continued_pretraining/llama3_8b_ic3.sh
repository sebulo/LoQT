torchrun --standalone --nproc_per_node 3 torchrun_main.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --dataset_name /data/scratch/gardar/gptsw3_training/raw_pretraining_data \
    --use_hf_model \
    --lr 0.0001 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 10000 \
    --batch_size 4 \
    --total_batch_size 504 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --dtype bfloat16 \
    --eval_every 2000 \
    --optimizer adam8bit \
    --use_mylora True \
    --lora_alpha 0.5 \
    --reset_adam_state False \
    --bnb_4bit_quant_type nf4 \
    --quantize_w '4bit' \
    --quantize_projection_matrix '4bit' \
    --joint_optim_iters 5 \
    --seed 42 \
    --save_every 1000 \
    --save_dir checkpoints/llama8b \
    --proj_gap_progression "exponential" \
    --increment_size 1.15 \
    --name llam3_8b_WP_Qloft_proj_gap1.15_100_icelandic \
    --use_offloading True 
    
#\
#    --use_eigenh_for_projection True
#     --single_gpu \
