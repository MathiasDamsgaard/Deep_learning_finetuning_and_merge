#!/bin/bash
#BSUB -J q_lora_plus_sweep
#BSUB -q gpuv100
#BSUB -W 24:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/logs/qlp_sweep%J.out
#BSUB -e hpc/logs/qlp_sweep%J.err

# Initialize Python environment
source .venv/bin/activate

# Run the Python script
python3 main.py --step sweep_manual --type Q_lora_plus