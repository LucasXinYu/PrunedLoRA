#!/bin/bash
export WANDB_MODE=disabled

get_free_port() {
    while true; do
        port=$(shuf -i 20000-30000 -n 1)
        if ! lsof -i:$port >/dev/null 2>&1; then
            echo $port
            return
        fi
    done
}

MASTER_PORT=$(get_free_port)
echo "Using free port: $MASTER_PORT"

output_dir=./checkpoints/fullft-llama3-8b
log_out=train_$(date +%Y%m%d_%H%M%S).log
mkdir -p $output_dir

nohup accelerate launch \
  --config_file fsdp_config.yaml \
  --main_process_port $MASTER_PORT \
  main_fft.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B \
  --dataset_name HuggingFaceH4/Code-Feedback \
  --output_dir $output_dir \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --max_steps 10000 \
  --learning_rate 1e-7 \
  --gradient_checkpointing True \
  > ${output_dir}/${log_out} 2>&1 &
