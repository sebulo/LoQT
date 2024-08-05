#!/bin/bash
python test_gsmk.py\
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --ckpt_dir checkpoints/meta-llama_Llama-2-7b-hf_GSMK_epochs5_seed987620240805_141153 \
  --use_loqt True \
  --batch_size 16 \
