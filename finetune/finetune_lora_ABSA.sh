# echo "sleep for 4 h"
# sleep 4h
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# export MODEL="internlm/internlm-xcomposer-vl-7b"
export MODEL="internlm/internlm-xcomposer2-vl-7b"
# export MODEL="/mnt/4dcc8983-1f7f-437e-bbca-b132b06be738/baoxiaoyi/ARRAUG2024/ASQP_rest16_predict/checkpoint-790"
# export DATA="data.txt"
export DATA="/home/baoxiaoyi/ACL_Arr_aug/final_data/twitter2015.txt"

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --img_size 490 \
    --given_num True \
    --bf16 True \
    --fix_vit True \
    --fix_sampler True \
    --use_lora True \
    --output_dir /mnt/4dcc8983-1f7f-437e-bbca-b132b06be738/baoxiaoyi/ARRAUG2024/twitter2015 \
    --num_train_epochs 30 \
    --batch_size 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 30 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --report_to "none" \
    --max_length 4096 \
    --deepspeed ds_config_zero2.json \
    --gradient_checkpointing True \
    --lora_alpha 128 \
