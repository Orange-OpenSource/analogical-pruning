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

import numpy
import pandas

import utils
from TqdmLoggingHandler import *


def main():
    start = time.time()

    parser = argparse.ArgumentParser(prog="decision_statistics", description="Compute statistics on decisions")
    parser.add_argument("--classifier-decisions", dest="classifier_decisions",
                        help="Pickle file containing decisions output by a classifier", required=True)
    parser.add_argument("--gold-decisions", dest="gold_decisions", help="CSV decision file", required=True)
    parser.add_argument("--output", dest="output", help="Output CSV file with statistics", required=True)
    args = parser.parse_args()

    # Logging parameters
    logger = logging.getLogger()
    tqdm_logging_handler = TqdmLoggingHandler()
    tqdm_logging_handler.setFormatter(logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(message)s"))
    logger.addHandler(tqdm_logging_handler)
    logger.setLevel(logging.INFO)

    # Loading decision files
    gold_decisions = pandas.read_csv(args.gold_decisions)
    gold_decisions.drop(gold_decisions[gold_decisions["from"] == gold_decisions["QID"]].index, inplace=True)
    classifier_decisions = pickle.load(open(args.classifier_decisions, "rb"))

    precisions = dict()  # Precision per fold
    recalls = dict()  # Recall per fold
    f1 = dict()  # F1 score per fold
    acc = dict() # Accuracy per fold

    for fold in classifier_decisions:

        fold_precision_num = 0
        fold_precision_den = 0
        fold_recall_num = 0
        fold_recall_den = 0
        fold_acc_num = 0
        fold_acc_den = 0

        for q in classifier_decisions[fold]:

            # Ground truth
            q_gold_kept = set(gold_decisions[(gold_decisions["from"] == q) & (gold_decisions["target"] == 1)]["QID"])
            q_gold_pruned = set(gold_decisions[(gold_decisions["from"] == q) & (gold_decisions["target"] == 0)]["QID"])

            q_classifier_pruned = {
                q2 for q2 in classifier_decisions[fold][q] if classifier_decisions[fold][q][q2]["decision"] == 0
            }
            q_classifier_kept = {
                q2 for q2 in classifier_decisions[fold][q] if classifier_decisions[fold][q][q2]["decision"] == 1
            }

            fold_precision_num += len(q_classifier_kept & q_gold_kept)
            fold_precision_den += len(q_classifier_kept)
            fold_recall_num += len(q_classifier_kept & q_gold_kept)
            fold_recall_den += len(q_gold_kept)
            fold_acc_num += len(q_classifier_kept & q_gold_kept) + len(q_classifier_pruned & q_gold_pruned)
            fold_acc_den += len(q_gold_kept) + len(q_gold_pruned)
           
        precisions[fold] = float(fold_precision_num) / (float(fold_precision_den))
        recalls[fold] = float(fold_recall_num) / (float(fold_recall_den))
        f1[fold] = 2 * (float(precisions[fold]) * float(recalls[fold])) \
                        / (float(precisions[fold]) + float(recalls[fold]))
        acc[fold] = float(fold_acc_num) / float(fold_acc_den)


    stats = pandas.DataFrame({
        "avg precision": [numpy.mean([precisions[fold] for fold in precisions])],
        "std precision": [numpy.std([precisions[fold] for fold in precisions])],
        "avg recall": [numpy.mean([recalls[fold] for fold in recalls])],
        "std recall": [numpy.std([recalls[fold] for fold in recalls])],
        "avg f1": [numpy.mean([f1[fold] for fold in f1])],
        "std f1": [numpy.std([f1[fold] for fold in f1])],
        "avg accuracy": [numpy.mean([acc[fold] for fold in acc])],
        "std accuracy": [numpy.std([acc[fold] for fold in acc])]
        })
    
    stats.to_csv(args.output, header=True, index=False)

    logger.info(f"Execution time = {utils.convert(time.time() - start)}")


if __name__ == '__main__':
    main()
