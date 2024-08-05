#!/bin/bash
python test_gsmk.py\
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --ckpt_dir checkpoints/meta-llama_Llama-2-7b-hf_GSMK_epochs6_seed1120240805_172950 \
  --use_loqt True \
  --batch_size 20 \

  #--ckpt_dir checkpoints/meta-llama_Llama-2-7b-hf_GSMK_epochs6_seed1120240805_172553 \
  #--ckpt_dir checkpoints/meta-llama_Llama-2-7b-hf_GSMK_epochs6_seed1120240805_173133 \

  #--ckpt_dir checkpoints/meta-llama_Llama-2-7b-hf_GSMK_epochs5_seed987620240805_141153 \