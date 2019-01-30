#!/bin/bash

dataset=$1
data_path=$2
dataset_type=$3

python train.py --data_path $data_path --dataset_type $dataset_type --save_dir ../data/${dataset}/random --split_type random --num_folds 13 --save_smiles_splits
python train.py --data_path $data_path --dataset_type $dataset_type --save_dir ../data/${dataset}/scaffold --split_type scaffold_balanced --num_folds 13 --save_smiles_splits