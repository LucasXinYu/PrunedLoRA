# PrunedLoRA: Robust Gradient-Based Structural Pruning for Low-rank Adaptation in Fine-tuning

This repository provides an implementation of **PrunedLoRA**, a robust gradient-based structural pruning method for low-rank adaptation in fine-tuning.  
We support multiple pruning baselines for comparison, including [Wanda](https://arxiv.org/abs/2306.11695), [SparseGPT](https://arxiv.org/abs/2301.00774), and [LLM-Pruner](https://github.com/horseee/LLM-Pruner).

---

## 1. Installation & Environment Setup

Clone this repository and set up the environment:

```bash
# Clone the repo
git clone https://github.com/yourname/prunedlora.git
cd prunedlora

# Create a conda environment
conda create -n prunedlora python=3.9 -y
conda activate prunedlora

# Install dependencies
pip install -r requirements.txt

# Install PEFT in editable mode
pip install -e peft

# Log in to Hugging Face for model and dataset access:
huggingface-cli login


## 2. Training

We support fine-tuning with different datasets and pruning strategies.

# Training datasets
# Option 1: MetaMathQA
dataset_name="meta-math/MetaMathQA"
dataset_config_name=None

# Option 2: Code-Feedback
dataset_name="HuggingFaceH4/Code-Feedback"
dataset_config_name="default"


## 3. Inference & Evaluation

We use[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation.


git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness

conda create -n lm-eval python=3.10 -y
conda activate lm-eval

pip install -e .

Run evaluation

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lm_eval \
  --model hf \
  --tasks humaneval \
  --model_args "pretrained=meta-llama/Meta-Llama-3-8B,tokenizer=meta-llama/Meta-Llama-3-8B,peft=/opt/tiger/mariana/llmeval/checkpoints/codefeedback/1e-5_alpha64_gradient_based_pruning_m2_64,dtype=bfloat16,parallelize=True" \
  --batch_size 8 \
  --output_path results/5e-5_adamw_pruning_256.json \
  --log_samples \
  --verbosity DEBUG \
  --confirm_run_unsafe_code




