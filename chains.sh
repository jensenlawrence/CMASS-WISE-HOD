#!/bin/bash
#SBATCH --account=def-wperciva
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=128G
#SBATCH --time=0-22:00
srun -n 4 python3 main.py --action mcmc