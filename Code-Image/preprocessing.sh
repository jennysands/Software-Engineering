#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o Preprocessin.%N.%j.out # STDOUT
#SBATCH -e Preprocessin.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate NMT

cd /home/u668954/Backup


python Birds.py
python Valid.py




#python Casper.py
#python model.py
#python training.py
