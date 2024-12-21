# README

## Overview

This repository contains code for fine-tuning and merging deep learning models using LoRA (Low-Rank Adaptation). The repository includes:

- Baseline model training and evaluation (baseline_model).
- LoRA-based model configuration, training, and evaluation (lora_manual, lora_model, etc.).
- Command-line usage through main.py with multiple configurable steps.

## Dependencies

- Python 3.11
- Torch 2.1.2
- Transformers
- BitsAndBytes
- PEFT
- WANDB
- And other libraries as listed in pyproject.toml or poetry.lock.

## Installation

1. Clone the repository.
2. From the repository root, install dependencies (for example, using Poetry):
   ```bash
   poetry install
   ```
3. Authenticate with Weights & Biases if needed:
   ```bash
   wandb login
   ```

## Usage

Most functionality is accessed through main.py with various steps:

1. Train a LoRA or baseline model:
   ```bash
   python main.py --step train --type lora --epochs 5
   ```
2. Perform a custom sweep with Weights & Biases:
   ```bash
   python main.py --step sweep_manual --type lora
   ```
3. Run the baseline model:
   ```bash
   python main.py --step BM --epochs 3
   ```
4. Print trainable parameters of a LoRA model:
   ```bash
   python main.py --step print_params --r 16 --lora_alpha 2.0 ...
   ```

Arguments like --learning_rate, --batch_size, etc., can be passed to customize training.

## File Structure (Key Parts)

- src/model/lora_manual.py: Contains the primary LoRATrainer class and get_lora_config function.
- src/config/model_config.py: Sets up different LoRA configurations (e.g., "lora", "Q_lora", etc.).
- main.py: Orchestrates model training, evaluation, and inference steps via CLI arguments.
