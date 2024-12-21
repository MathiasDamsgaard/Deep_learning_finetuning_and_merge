#!/bin/bash
#BSUB -J q_lora
#BSUB -q gpuv100
#BSUB -W 08:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/logs/q_lora%J.out
#BSUB -e hpc/logs/q_lora%J.err

# Initialize Python environment
source .venv/bin/activate

python3 main.py --num_folds 1 --step train --type Q_lora --learning_rate 0.00402017718214621 --lora_alpha 3.250647057823948 --r 32
