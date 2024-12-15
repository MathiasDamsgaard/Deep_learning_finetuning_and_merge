#!/bin/bash
#BSUB -J baseline
#BSUB -q gpuv100
#BSUB -W 08:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/logs/baseline%J.out
#BSUB -e hpc/logs/baseline%J.err

# Initialize Python environment
source .venv/bin/activate

python3 main.py --step train --epochs 15 --num_folds 5 --type baseline --batch_size 16 --learning_rate 0.002