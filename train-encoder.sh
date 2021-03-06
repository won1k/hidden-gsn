#!/bin/bash
#SBATCH -n 12                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 2-23:00              # Runtime in D-HH:MM
#SBATCH -p holyseasgpu       # Partition to submit to
#SBATCH --mem=20000               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out/ae_2.out      # File to which STDOUT will be written
#SBATCH -e dump/ae_2.err      # File to which STDERR will be written
#SBATCH --gres=gpu:1

th train-encoder.lua -epochs 100 -savefile checkpoint/ptb-ae-noseq
