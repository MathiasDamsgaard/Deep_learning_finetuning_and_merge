#!/bin/bash
#BSUB -J lora_plus
#BSUB -q gpuv100
#BSUB -W 08:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/logs/lora_plus_train%J.out
#BSUB -e hpc/logs/lora_plus_train%J.err

# Initialize Python environment
source .venv/bin/activate

python3 main.py --num_folds 1 --step train --type lora_plus --learning_rate 0.0002 --loraplus_lr_ratio 30 --lora_alpha 5.7806047903277 --r 32
