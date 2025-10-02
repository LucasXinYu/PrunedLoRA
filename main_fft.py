#!/usr/bin/env python
# coding=utf-8

import logging
import math
import os
import sys
from accelerate import Accelerator
import torch
import datasets
from datasets import load_dataset
import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers import get_scheduler
from transformers.trainer_utils import get_last_checkpoint
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import numpy as np
from tqdm import tqdm
import re


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=False)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    block_size: Optional[int] = field(default=1024)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=10)
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator()
    logging.basicConfig(level=logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    set_seed(training_args.seed)

    # ===== Load dataset =====
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if "train_sft" in raw_datasets and "test_sft" in raw_datasets:
            raw_datasets["train"] = raw_datasets["train_sft"]
            raw_datasets["validation"] = raw_datasets["test_sft"]
        elif "validation" not in raw_datasets:
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    else:
        raise ValueError("You must provide a dataset_name or train/validation file.")

    # ===== Load config, tokenizer, model =====
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer)

    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        cache_dir=model_args.cache_dir,
    )

    if accelerator.is_local_main_process:
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(model)

    # ===== Tokenization =====
    column_names = list(raw_datasets["train"].features)

    def tokenize_function(examples):
        #print("==== Raw examples before tokenization ====")
        #for key, value in examples.items():
        #    print(f"[sample example]: {key}: {value[:1]}")
        #print("==========================================")

        dataset_name = data_args.dataset_name.lower()
        prompts = []
        outputs = []

        if "meta-math" in dataset_name or "gsm8k" in dataset_name:
            # === math ===
            prompts = [f"Question: {q} </s> Answer: " for q in examples["query"]]
            outputs = examples["response"]

        elif "humaneval" in dataset_name:
            # === code ===
            prompts = [p for p in examples["prompt"]]
            outputs = examples["canonical_solution"]

        elif "code-feedback" in dataset_name:
            # === dialogue/code feedback ===
            for msgs in examples["messages"]:
                if len(msgs) < 2:
                    continue
                prompt = ""
                for m in msgs[:-1]:
                    if m["role"] in ["user", "system"]:
                        prompt += f"{m['role'].capitalize()}: {m['content']}\n"
                prompts.append(prompt)
                outputs.append(msgs[-1]["content"])
            print("tokenizer finished - code-feedback")

        else:
            # === generic dataset with "text" column ===
            full_texts = examples["text"]
            tokenized = tokenizer(
                full_texts,
                truncation=True,
                max_length=data_args.block_size,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        # === For prompt + output style datasets ===
        full_texts = [p + o for p, o in zip(prompts, outputs)]
        tokenized_full = tokenizer(full_texts, truncation=True, max_length=data_args.block_size)
        tokenized_prompts = tokenizer(prompts, truncation=True, max_length=data_args.block_size)

        labels = [ids.copy() for ids in tokenized_full["input_ids"]]
        for i, prompt_ids in enumerate(tokenized_prompts["input_ids"]):
            prompt_length = len(prompt_ids)
            labels[i][:prompt_length] = [-100] * prompt_length  # mask prompt loss
        tokenized_full["labels"] = labels

        return tokenized_full

    with training_args.main_process_first(desc="tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing dataset",
        )

    # ===== Group texts =====
    block_size = data_args.block_size
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # ===== Dataloaders =====
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=training_args.per_device_eval_batch_size
    )

    # ===== Optimizer =====
    # ===== Optimizer =====
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=training_args.learning_rate,
    weight_decay=training_args.weight_decay,
    betas=(0.9, 0.95),  
    eps=1e-8            
    )

    # ===== Scheduler =====
    num_update_steps_per_epoch = len(train_dataloader)
    if training_args.max_steps > 0:
        max_training_steps = training_args.max_steps
    else:
        max_training_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="cosine",  
        optimizer=optimizer,
        num_warmup_steps=int(0.03 * max_training_steps),  
        num_training_steps=max_training_steps,
    )


    # ===== Accelerator prepare =====
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )


    # ===== Training loop =====
    model.train()
    global_step = 0
    max_steps = training_args.max_steps if training_args.max_steps > 0 else None

    training_args.num_train_epochs = 10

    for epoch in range(int(training_args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            grad_norm = None
            total_norm = 0.0
            parameters = [p for p in model.parameters() if p.grad is not None]
            if len(parameters) > 0:
                total_norm = torch.norm(
                    torch.stack([
                        torch.norm(p.grad.detach(), 2).to(torch.float32)
                        for p in parameters
                    ]),
                    2
                ).item()
                grad_norm = total_norm

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if accelerator.is_local_main_process and global_step % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                if grad_norm is not None:
                    print(f"Epoch {epoch}, Step {global_step}, "
                          f"Loss {loss.item():.4f}, "
                          f"GradNorm {grad_norm:.4f}, "
                          f"LR {current_lr:.6e}")
                else:
                    print(f"Epoch {epoch}, Step {global_step}, "
                          f"Loss {loss.item():.4f}, "
                          f"LR {current_lr:.6e}")

            if max_steps is not None and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}, stopping training.")
                break

    # ===== Save =====
    accelerator.wait_for_everyone()   # <— make sure all ranks finished training steps
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(training_args.output_dir,
                                        save_function=accelerator.save)
        tokenizer.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()   # <— make sure non-zero ranks wait for rank0 to finish IO

    # Optional: free GPU memory before NCCL teardown
    accelerator.free_memory()

    # Tell Accelerate you're done (orders the shutdown)
    accelerator.end_training()

if __name__ == "__main__":
    main()

