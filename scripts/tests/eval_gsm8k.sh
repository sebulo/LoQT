#!/bin/bash
python eval_gsmk.py\
  --model_name_or_path EleutherAI/gpt-neo-1.3B \
  --ckpt_dir checkpoints/gpt-neo-1.3B \
  --batch_size 16
