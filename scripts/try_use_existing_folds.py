from argparse import ArgumentParser
import os
import pickle
import numpy as np

def replace_folds(args):
    if args.num_test_folds == 3 and args.val_folds_per_test == 1:
        args.dataset_size = 'big'
    elif args.num_test_folds == 10 and args.val_folds_per_test == 1:
        args.dataset_size = 'one'
    elif args.num_test_folds == 10 and args.val_folds_per_test == 3:
        args.dataset_size = 'small'
    else:
        raise ValueError
    crossval_index_path = os.path.join(args.crossval_index_path, args.dataset_size)
    fold_index = 0
    used_some_folds = False
    for i in range(10):
        if os.path.exists(os.path.join(crossval_index_path, str(i) + '_opt.pkl')):
            used_some_folds = True
            with open(os.path.join(crossval_index_path, str(i) + '_opt.pkl'), 'rb') as f:
                fold_indices = pickle.load(f)
                all_splits = []
                all_opt_splits = []
                for array in fold_indices:
                    splits = []
                    for split in array:
                        current_split = []
                        for index in split:
                            with open(os.path.join(args.crossval_folds_path, args.split_type, str(index) + '.pkl'), 'rb') as f:
                                indices = pickle.load(f)
                                current_split.append(indices)
                        current_split = np.concatenate(current_split)
                        splits.append(current_split)
                    assert len(splits) == 3
                    all_splits.append(splits)
                    all_opt_splits.append([splits[0], splits[1], splits[1]]) # train, val, val
            assert len(all_splits) == args.val_folds_per_test
            for j in range(len(all_splits)):
                os.makedirs(os.path.join(args.save_dir, args.split_type, f'fold_{fold_index}', f'{j}'), exist_ok=True)
                with open(os.path.join(args.save_dir, args.split_type, f'fold_{fold_index}', f'{j}', 'split_indices.pckl'), 'wb') as wf:
                    pickle.dump(all_splits[j], wf)
            with open(os.path.join(args.save_dir, args.split_type, f'fold_{fold_index}', 'opt_split_indices.pckl'), 'wb') as wf:
                pickle.dump(all_opt_splits, wf)
            fold_index += 1
    print('used existing folds:', used_some_folds)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file with dataset of molecules')
    parser.add_argument('--crossval_index_path', type=str, required=True,
                        help='Path to dir of crossval indices')
    parser.add_argument('--crossval_folds_path', type=str, required=True,
                        help='Path to dir of crossval folds')
    parser.add_argument('--split_type', type=str, required=True, choices=['random', 'scaffold'],
                        help='split type')
    parser.add_argument('--val_folds_per_test', type=int, required=True,
                        help='val folds per test')
    parser.add_argument('--num_test_folds', type=int, required=True,
                        help='num test folds')
    parser.add_argument('--save_dir', type=str, 
                        help='Path to save dir')
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.data_path)
    
    replace_folds(args)
