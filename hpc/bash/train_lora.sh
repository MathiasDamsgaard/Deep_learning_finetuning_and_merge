#!/bin/bash
#BSUB -J lora
#BSUB -q gpuv100
#BSUB -W 08:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/logs/lora_train%J.out
#BSUB -e hpc/logs/lora_train%J.err

# Initialize Python environment
source .venv/bin/activate

python3 main.py --step train --type lora --num_folds 1
