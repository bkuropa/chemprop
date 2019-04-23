from argparse import ArgumentParser
import pickle
import os
import random
from copy import deepcopy

import numpy as np

from chemprop.data.utils import get_data
from chemprop.data.scaffold import scaffold_to_smiles

def split_indices(all_indices, scaffold=False, data=None, shuffle=True):
    num_data = len(all_indices)
    if scaffold:
        scaffold_to_indices = scaffold_to_smiles(data.mols(), use_indices=True)
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)
        fold_indices = [[] for _ in range(args.num_folds)]
        for s in index_sets:
            length_array = [len(fi) for fi in fold_indices]
            min_index = length_array.index(min(length_array))
            fold_indices[min_index] += s
    else: # random
        if shuffle:
            random.shuffle(all_indices)
        fold_indices = []
        for i in range(args.num_folds):
            begin, end = int(i * num_data / args.num_folds), int((i+1) * num_data / args.num_folds)
            fold_indices.append(np.array(all_indices[begin:end]))
    return fold_indices

def create_time_splits(args):
    #ASSUME DATA GIVEN IN CHRONOLOGICAL ORDER. 
    # this will dump a very different format of indices, with all in one file; TODO modify as convenient later.
    data = get_data(args.data_path)
    num_data = len(data)
    all_indices = list(range(num_data))
    fold_indices = {'random':[], 'scaffold':[], 'time':[]}
    for i in range(args.num_folds - args.time_folds_per_train_set - 1):
        begin, end = int(i * num_data / args.num_folds), int((i + args.time_folds_per_train_set + 2) * num_data / args.num_folds)
        subset_indices = all_indices[begin:end]
        subset_data = data[begin:end] # TODO check this syntax?
        fold_indices['random'].append(split_indices(deepcopy(subset_indices)))
        fold_indices['scaffold'].append(split_indices(subset_indices, scaffold=True, data=subset_data))
        fold_indices['time'].append(split_indices(subset_data, shuffle=False))
    for split_type in ['random', 'scaffold', 'time']:
        for i in range(len(fold_indices[split_type])):
            with open(os.path.join(args.save_dir, split_type, 'fold_' + str(i), 'split_indices.pckl'), 'wb') as wf:
                pickle.dump(fold_indices[split_type][i], wf) # each is a pickle file containing a list of length-3 index lists for train/val/test

def create_crossval_splits(args):
    data = get_data(args.data_path)
    num_data = len(data)
    if args.split_type == 'random':
        all_indices = list(range(num_data))
        fold_indices = split_indices(all_indices, scaffold=False)
    elif args.split_type == 'scaffold':
        all_indices = list(range(num_data))
        fold_indices = split_indices(all_indices, scaffold=True, data=data)
    else:
        raise ValueError
    random.shuffle(fold_indices)
    for i in range(args.test_folds_to_test):
        for j in range(args.val_folds_per_test):
            os.makedirs(os.path.join(args.save_dir, args.split_type, f'fold_{i}', f'{j}'), exist_ok=True)
            with open(os.path.join(args.save_dir, args.split_type, f'fold_{i}', f'{j}', 'split_indices.pckl'), 'wb') as wf:
                val_idx = (i+j) % args.num_folds
                val = fold_indices[val_idx]
                test = fold_indices[i]
                train = []
                for k in range(args.num_folds):
                    if k != i and k != val_idx:
                        train.append(fold_indices[k])
                train = np.concatenate(train)
                pickle.dump([train, val, test], wf)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file with dataset of molecules')
    parser.add_argument('--save_dir', type=str,
                        help='Path to CSV file where splits will be saved')
    parser.add_argument('--split_type', type=str, choices=['random', 'scaffold', 'time_window'], required=True,
                        help='Random or scaffold based split')
    parser.add_argument('--num_folds', type=int, default=10,
                        help='Number of cross validation folds')
    parser.add_argument('--test_folds_to_test', type=int, default=3,
                        help='Number of test folds')  
    parser.add_argument('--val_folds_per_test', type=int, default=3,
                        help='Number of val folds')                    
    parser.add_argument('--time_folds_per_train_set', type=int, default=3,
                        help='X:1:1 train:val:test for time split sliding window')
    args = parser.parse_args()

    random.seed(0)

    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.data_path)
    
    if args.split_type == 'time_window':
        create_time_splits(args)
    else:
        create_crossval_splits(args)
