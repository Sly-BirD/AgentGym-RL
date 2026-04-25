#!/bin/bash
# ==============================================================================
# Cloud GPU Setup Script for RL-WebNav Agent
# ==============================================================================
# This script sets up the complete training environment on a fresh Linux GPU instance.
#
# Tested on: Lightning.ai (A100 40GB), RunPod (A100 80GB)
# Requirements: NVIDIA GPU (24GB+ VRAM), CUDA 12.4, Ubuntu 22.04+
#
# Usage:
#   chmod +x scripts/cloud_setup.sh
#   bash scripts/cloud_setup.sh
#
# What this script does (step by step):
#
# 1. CONDA: Creates an isolated Python 3.10 environment called "agentgym-rl"
#    → Why? Different projects need different Python/library versions.
#      Conda prevents them from conflicting.
#
# 2. PYTORCH 2.4 + CUDA 12.4: Installs the deep learning framework.
#    → PyTorch is the engine that runs neural networks on GPUs.
#    → CUDA 12.4 is NVIDIA's GPU computing toolkit version.
#
# 3. FLASH ATTENTION 2.7: Installs optimized attention mechanism.
#    → Attention is the core operation in Transformers/LLMs.
#    → Flash Attention makes it 2-4x faster and uses 5-20x less memory.
#    → Critical for long sequences (web pages can be thousands of tokens).
#
# 4. AGENTGYM-RL: Installs the RL training engine (verl + multi-turn RL).
#    → This is the core framework that runs GRPO/PPO training.
#
# 5. AGENTGYM: Installs the environment client (connects to WebArena etc.)
#    → This is the "world" the agent interacts with.
#
# 6. MODEL: Downloads Qwen2.5-7B-Instruct from HuggingFace.
#    → This is our starting LLM policy (7 billion parameters).
#
# 7. DATA: Downloads training task IDs from HuggingFace.
#    → These are task definitions (not full trajectories).
#
# ==============================================================================

set -e  # Exit on any error
set -x  # Print each command before executing (for debugging)

echo "============================================="
echo "  RL-WebNav Agent — Cloud Setup"
echo "============================================="
echo ""

# --------------------------------------------------
# Step 0: Verify GPU is available
# --------------------------------------------------
echo ">>> Step 0: Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is NVIDIA driver installed?"
    exit 1
fi
nvidia-smi
echo ""

# Check CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "Detected CUDA Version: ${CUDA_VERSION}"
echo ""

# --------------------------------------------------
# Step 1: Create Conda environment
# --------------------------------------------------
# Conda = package manager that creates isolated environments.
# "agentgym-rl" = the name of our environment.
# python==3.10 = the exact Python version the repo requires.
# --------------------------------------------------
echo ">>> Step 1: Creating conda environment..."
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init
fi

conda create -n agentgym-rl python==3.10 -y
conda activate agentgym-rl
echo ""

# --------------------------------------------------
# Step 2: Install PyTorch 2.4 with CUDA 12.4
# --------------------------------------------------
# PyTorch = the deep learning framework (like TensorFlow, but more research-friendly).
# cu124 = compiled against CUDA 12.4 (must match your driver).
# --------------------------------------------------
echo ">>> Step 2: Installing PyTorch 2.4 + CUDA 12.4..."
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
echo ""

# Verify PyTorch can see the GPU
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
echo ""

# --------------------------------------------------
# Step 3: Install Flash Attention 2.7
# --------------------------------------------------
# Flash Attention = optimized GPU kernel for the attention operation.
# Regular attention: O(N²) memory for sequence length N.
# Flash Attention: O(N) memory — enables much longer sequences.
# Pre-built wheel = compiled binary (avoids 30+ min compilation).
# --------------------------------------------------
echo ">>> Step 3: Installing Flash Attention..."
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
pip3 install $FLASH_ATTENTION_NAME
rm -f $FLASH_ATTENTION_NAME
echo ""

# --------------------------------------------------
# Step 4: Install AgentGym-RL (Training Engine)
# --------------------------------------------------
# "pip install -e ." = editable install.
# -e means "editable" — Python links to the source code directly.
# So if you modify any .py file, the changes take effect immediately
# without needing to reinstall.
#
# This installs Verl (RL trainer from Bytedance) + our agent training code:
#   - GRPO, PPO, REINFORCE++ algorithms
#   - Multi-turn rollout handler
#   - Environment client
#   - FSDP distributed training support
# --------------------------------------------------
echo ">>> Step 4: Installing AgentGym-RL..."
cd AgentGym-RL
pip3 install -e .
cd ..
echo ""

# --------------------------------------------------
# Step 5: Install AgentGym (Environment Module)
# --------------------------------------------------
# AgentGym = the collection of environments (WebArena, TextCraft, etc.)
# It provides the client-side code that talks to environment servers via HTTP.
#
# Architecture:
#   [Environment Server (Docker)] <--HTTP--> [EnvClient (Python)]
#   The server runs the actual web apps / games.
#   The client sends actions (click, type) and receives observations (page HTML).
# --------------------------------------------------
echo ">>> Step 5: Installing AgentGym..."
cd AgentGym/agentenv
pip3 install -e .
pip3 install transformers==4.51.3
cd ../..
echo ""

# --------------------------------------------------
# Step 6: Install additional dependencies
# --------------------------------------------------
echo ">>> Step 6: Installing additional dependencies..."
pip3 install gradio>=4.0.0 plotly pandas wandb
echo ""

# --------------------------------------------------
# Step 7: Download the base model (Qwen2.5-7B-Instruct)
# --------------------------------------------------
# Qwen2.5-7B-Instruct = a 7 billion parameter language model.
# "Instruct" = fine-tuned to follow instructions (vs. raw "base" model).
# This is our STARTING POLICY — the LLM brain before RL training.
# Size: ~14 GB in bfloat16 (half-precision floating point).
#
# HuggingFace = a platform for sharing ML models (like GitHub for AI models).
# --------------------------------------------------
echo ">>> Step 7: Downloading Qwen2.5-7B-Instruct..."
mkdir -p models
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    local_dir='models/Qwen2.5-7B-Instruct',
    local_dir_use_symlinks=False
)
print('Model downloaded successfully!')
"
echo ""

# --------------------------------------------------
# Step 8: Download training data
# --------------------------------------------------
# AgentGym-RL-Data-ID = task IDs for training.
# NOT full trajectories — just identifiers like "webarena_task_001".
# During training, these IDs are sent to the environment server
# which initializes the corresponding task.
# --------------------------------------------------
echo ">>> Step 8: Downloading training data..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='AgentGym/AgentGym-RL-Data-ID',
    repo_type='dataset',
    local_dir='AgentItemId',
    local_dir_use_symlinks=False
)
print('Training data downloaded successfully!')
"
echo ""

# --------------------------------------------------
# Step 9: Verify installation
# --------------------------------------------------
echo ">>> Step 9: Verifying installation..."
python3 -c "
import torch
import verl
import transformers

print('=== Installation Verification ===')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'Transformers: {transformers.__version__}')
print('================================')
print('All good! Ready to train.')
"
echo ""

echo "============================================="
echo "  Setup Complete!"
echo ""
echo "  Next steps:"
echo "  1. Launch WebArena environment server (Docker)"
echo "  2. Run baseline eval:  bash examples/eval/webarena_eval.sh"
echo "  3. Start training:     bash examples/train/AgentGym-RL/webarena_train.sh"
echo "============================================="
