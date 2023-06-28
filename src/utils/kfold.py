"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import argparse
import pickle
import time

import pandas
from sklearn.model_selection import KFold

import utils


def main():
    start = time.time()

    parser = argparse.ArgumentParser(prog="kfold", description="K-folding of labeled decisions dataset")
    parser.add_argument("--nb-fold", dest="nb_fold", help="Number of folds", required=True, type=int)
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--output", dest="output", help="Output pickle hashmap", required=True)
    args = parser.parse_args()

    folds = dict()

    # Load starting QIDs from decisions
    df = pandas.read_csv(args.decisions)
    starting_qids = list(set(df['from']))

    # KFold definition
    kf = KFold(n_splits=args.nb_fold, shuffle=True)
    for i, (train_set, test_test) in enumerate(kf.split(starting_qids)):
        folds[i] = {
            "train": [starting_qids[i] for i in train_set],
            "test": [starting_qids[i] for i in test_test]
        }

    pickle.dump(folds, open(args.output, "wb"))

    print("Execution time = ", utils.convert(time.time() - start))


if __name__ == '__main__':
    main()
