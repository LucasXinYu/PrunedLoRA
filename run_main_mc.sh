#!/bin/bash
export WANDB_MODE=disabled

optim_notes=prunedlora
split_strategy=iid
num_rounds=2
num_clients=1
sample_clients=1
block_size=512
fp16=True
lr_scheduler_type="cosine"
max_gate_samples=50
max_train_samples=100000
do_train=True

# dataset_name = "meta-math/MetaMathQA"
# dataset_config_name = None


model_name_or_path="meta-llama/Meta-Llama-3-8B"
tokenizer_name="meta-llama/Meta-Llama-3-8B"

dataset_name=HuggingFaceH4/Code-Feedback
dataset_config_name=default 

per_device_train_batch_size=4
per_device_eval_batch_size=1
gradient_accumulation_steps=8
gradient_checkpointing=False
max_steps=500000
dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=no
seed=42
lora_rank=64

log_out=log.out
learning_rates=(1e-4)
lora_alphas=(128)

NPROC_PER_NODE=8
get_free_port() {
    while true; do
        port=$(shuf -i 20000-30000 -n 1)
        if ! lsof -i:$port >/dev/null 2>&1; then
            echo $port
            return
        fi
    done
}

for learning_rate in "${learning_rates[@]}"; do
    for lora_alpha in "${lora_alphas[@]}"; do

        output_dir=./checkpoints/${model_name}/codefeedback/${learning_rate}_alpha${lora_alpha}_${optim_notes}_${lora_rank}_${max_steps}
        echo "Saving to ${output_dir}"
        mkdir -p ${output_dir}

        echo  --model_name_or_path ${model_name_or_path} \
              --tokenizer_name ${tokenizer_name} \
              --output_dir ${output_dir} \
              --dataset_name ${dataset_name} \
              --dataset_config_name ${dataset_config_name} \
              --per_device_train_batch_size ${per_device_train_batch_size} \
              --per_device_eval_batch_size ${per_device_eval_batch_size} \
              --max_steps ${max_steps} \
              --overwrite_output_dir \
              --do_train ${do_train} \
              --do_eval \
              --lr_scheduler_type ${lr_scheduler_type} \
              --block_size ${block_size} \
              --seed ${seed} \
              --fp16 ${fp16} \
              --lora_rank ${lora_rank} \
              --lora_alpha ${lora_alpha} \
              --gradient_checkpointing ${gradient_checkpointing} \
              --dataloader_num_workers ${dataloader_num_workers} \
              --disable_tqdm False \
              --learning_rate ${learning_rate} \
              --optim_notes ${optim_notes} \
              --split_strategy ${split_strategy} \
              --num_rounds ${num_rounds} \
              --num_clients ${num_clients} \
              --sample_clients ${sample_clients} \
              --max_gate_samples ${max_gate_samples} \
              --max_train_samples ${max_train_samples} \
              --gradient_accumulation_steps ${gradient_accumulation_steps} \
              > ${output_dir}/config.txt

        if [ -f ${output_dir}/${log_out} ]; then
            rm -f ${output_dir}/${log_out}
        fi

        export MASTER_PORT=$(get_free_port)

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
            nohup accelerate launch \
            --mixed_precision bf16 \
            --num_processes 8 \
            --num_machines 1 \
            --multi_gpu \
            --main_process_port $MASTER_PORT \
            main_lora_pruning.py \
            --model_name_or_path ${model_name_or_path} \
            --tokenizer_name ${tokenizer_name} \
            --use_auth_token False \
            --output_dir ${output_dir} \
            --dataset_name ${dataset_name} \
            --dataset_config_name ${dataset_config_name} \
            --per_device_train_batch_size ${per_device_train_batch_size} \
            --per_device_eval_batch_size ${per_device_eval_batch_size} \
            --max_steps ${max_steps} \
            --overwrite_output_dir \
            --do_train ${do_train} \
            --do_eval \
            --lr_scheduler_type ${lr_scheduler_type} \
            --seed ${seed} \
            --fp16 ${fp16} \
            --lora_rank ${lora_rank} \
            --lora_alpha ${lora_alpha} \
            --gradient_checkpointing ${gradient_checkpointing} \
            --block_size ${block_size} \
            --dataloader_num_workers ${dataloader_num_workers} \
            --disable_tqdm False \
            --learning_rate ${learning_rate} \
            --optim_notes ${optim_notes} \
            --split_strategy ${split_strategy} \
            --num_rounds ${num_rounds} \
            --num_clients ${num_clients} \
            --sample_clients ${sample_clients} \
            --max_gate_samples ${max_gate_samples} \
            --max_train_samples ${max_train_samples} \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            > ${output_dir}/${log_out} 2>&1 &

    done
done
