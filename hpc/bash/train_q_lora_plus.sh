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

python3 main.py --step train --epochs 15 --batch_size 16 --num_folds 5 --type Q_lora_plus --learning_rate 1e-5 --dropout 0.2 --r 128 --loraplus_lr_ratio 16