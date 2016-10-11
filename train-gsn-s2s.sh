#!/bin/bash
#SBATCH -n 12                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 0-10:00              # Runtime in D-HH:MM
#SBATCH -p holyseasgpu       # Partition to submit to
#SBATCH --mem=10000               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out/gsn-s2s.out      # File to which STDOUT will be written
#SBATCH -e dump/gsn-s2s.err      # File to which STDERR will be written
#SBATCH --gres=gpu:1

#~wangalexc/torch/install/bin/th train-gsn-s2s.lua
th train-gsn-s2s.lua
