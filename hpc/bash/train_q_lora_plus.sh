#!/bin/bash
#BSUB -J lora_q_plus
#BSUB -q gpuv100
#BSUB -W 08:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/logs/lora_q_plus_train%J.out
#BSUB -e hpc/logs/lora_q_plus_train%J.err

# Initialize Python environment
source .venv/bin/activate

python3 main.py --num_folds 1 --step train --type Q_lora_plus --learning_rate 3e-4 --r 64 --loraplus_lr_ratio 32
