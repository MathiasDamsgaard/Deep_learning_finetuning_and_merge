#!/bin/bash
#BSUB -J q_lora_train
#BSUB -q gpuv100
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/logs/q_lora%J.out
#BSUB -e hpc/logs/q_lora%J.err

# Initialize Python environment
source .venv/bin/activate

main.py --step train --epochs 50 --type Q_lora --learning_rate 5e-5 --dropout 0.2 --r 32