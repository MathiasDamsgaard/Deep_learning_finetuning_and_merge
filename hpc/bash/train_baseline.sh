#!/bin/bash
#BSUB -J q_lora_baseline_train
#BSUB -q gpuv100
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/logs/q_lora%J.out
#BSUB -e hpc/logs/q_lora%J.err

# Initialize Python environment
source .venv/bin/activate

python3 main.py --step none --BM True --epochs 30