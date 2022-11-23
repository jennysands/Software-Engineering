#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o Classifier.%N.%j.out # STDOUT
#SBATCH -e Classifier.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate NMT

cd /home/u668954/Backup



python GUI.py
#python Testing.py
