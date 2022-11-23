#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o CSVcreate.%N.%j.out # STDOUT
#SBATCH -e CSVcreate.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate NMT

cd /home/u668954

python create_bird.py
python create_insect.py
python append_df.py
