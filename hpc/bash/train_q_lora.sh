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

python3 main.py --step train --epochs 15 --num_folds 5 --type Q_lora --batch_size 16 --learning_rate 0.002574838678975716 --lora_alpha 5.163468946270533 --dropout 0.2 --r 32