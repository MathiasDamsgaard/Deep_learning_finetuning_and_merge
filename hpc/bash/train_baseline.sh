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

python3 main.py --num_folds 1 --step train --type baseline --learning_rate 0.00013631447543921584
