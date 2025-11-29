
pip install ms_swift==3.9.0  # Swift framework for efficient model training
pip show ms_swift
pip install liger_kernel
pip show liger_kernel==0.5.2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # Enable CUDA memory optimization
export INFONCE_USE_BATCH=False
export TENSORBOARD_DIR="./logs/tensorboard"


output_dir="../models/qwen25_32b_lora_groupwise_reranker_sft"

NNODES=${WORLD_SIZE:-1}

NPROC_PER_NODE=8 \
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} \
MASTER_PORT=${MASTER_PORT:-12346} \
NNODES=$NNODES \
NODE_RANK=${RANK:-0} \
swift sft \
    --model ../models/Qwen2.5-32B-Instruct \
    --train_type lora \
    --model_type qwen2_5 \
    --dataset "./groupwise_sft_data_example.jsonl" \
    --split_dataset_ratio 0 \
    --save_strategy epoch \
    --learning_rate 3e-5 \
    --torch_dtype bfloat16 \
    --output_dir ${output_dir} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last true \
    --save_only_model true \
    --deepspeed zero3 \
    --max_new_tokens 8192 \
    --load_from_cache_file False \
    --max_length 32000 \
    --dataset_num_proc 64 \
    --attn_impl flash_attn \
    --packing true \
    --use_liger_kernel true