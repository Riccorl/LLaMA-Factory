#!/bin/bash

python scripts/vllm_infer.py \
    --model_name_or_path Qwen/Qwen3.5-4B \
    --dataset risk_dev_no_desc_700 \
    --template qwen3_5_nothink \
    --cutoff_len 8192 \
    --save_name predictions/qwen3_5-4b/risk_dev_no_desc_700_inference_results_no_thinking.json \
    --temperature 1.0 \
    --max_new_tokens 8192 \
    --enable_thinking False \
    --batch_size 4096 \
    --top_p 0.95 \
    --top_k 20 \
    --min_p 0.0 \
    --presence_penalty 1.5 \
    --repetition_penalty 1.0

#saves/Qwen3-4B-Instruct-2507/lora/train_2026-06-04-11-40-no_desc_700_dropped/merged \

# qwen 3.5
# Instruct (or non-thinking) mode for reasoning tasks: 
# temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
# Thinking mode for general tasks: 
# temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0