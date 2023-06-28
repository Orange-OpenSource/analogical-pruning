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
import lmdb
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
    parser.add_argument("--folds", dest="folds_path", help="File containing folds", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
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

    # Load folds
    folds = pickle.load(open(args.folds_path, "rb"))

    # Load Wikidata hashmap (QID -> properties)
    dump = lmdb.open(args.wikidata_hashmap, readonly=True, readahead=False)
    wikidata_hashmap = dump.begin()

    my_stats = dict()

    precisions = dict()  # Precision per fold
    recalls = dict()  # Recall per fold
    f1 = dict()  # F1 score per fold
    acc = dict() # Accuracy per fold

    precisions_unseen = dict()  # Precision per fold
    recalls_unseen = dict()  # Recall per fold
    f1_unseen = dict()  # F1 score per fold
    acc_unseen = dict() # Accuracy per fold

    precisions_seen = dict()  # Precision per fold
    recalls_seen = dict()  # Recall per fold
    f1_seen = dict()  # F1 score per fold
    acc_seen = dict() # Accuracy per fold

    for fold in classifier_decisions:

        val_set = set(folds[(fold+1)%5]["test"])

        # Train model
        train_set = set(folds[fold]["train"]) - val_set

        compare = set(gold_decisions[gold_decisions["from"].isin(train_set)]["QID"])
        compare |= set(gold_decisions[gold_decisions["from"].isin(train_set)]["from"])

        fold_precision_num = 0
        fold_precision_den = 0
        fold_recall_num = 0
        fold_recall_den = 0
        fold_acc_num = 0
        fold_acc_den = 0

        # Unseen
        fold_precision_num_unseen = 0
        fold_precision_den_unseen = 0
        fold_recall_num_unseen = 0
        fold_recall_den_unseen = 0
        fold_acc_num_unseen = 0
        fold_acc_den_unseen = 0

        # Seen
        fold_precision_num_seen = 0
        fold_precision_den_seen = 0
        fold_recall_num_seen = 0
        fold_recall_den_seen = 0
        fold_acc_num_seen = 0
        fold_acc_den_seen = 0

        unseen_counter = 0
        seen_counter = 0

        for q in classifier_decisions[fold]:

            seen_cl = set()
            unseen_cl = set()

            seen_classes = {q}

            q_adjacency = utils.get_hashmap_content(q, wikidata_hashmap)

            q_gold_kept_seen = set()
            q_gold_kept_unseen = set()
            q_gold_pruned_seen = set()
            q_gold_pruned_unseen = set()

            # Ground truth
            q_gold_kept = set(gold_decisions[(gold_decisions["from"] == q) & (gold_decisions["target"] == 1)]["QID"])
            q_gold_pruned = set(gold_decisions[(gold_decisions["from"] == q) & (gold_decisions["target"] == 0)]["QID"])

            reached_q_w_decisions = q_gold_kept | q_gold_pruned

            if q_adjacency is not None and ("P31" in q_adjacency["claims"] or "P279" in q_adjacency["claims"] or
                                "(-)P279" in q_adjacency["claims"]):
                    
                classes = set()

                if "P31" in q_adjacency["claims"]:
                    classes |= utils.get_relation_objects(q_adjacency["claims"]["P31"]) & reached_q_w_decisions
                if "P279" in q_adjacency["claims"]:
                    classes |= utils.get_relation_objects(q_adjacency["claims"]["P279"]) & reached_q_w_decisions
                if "(-)P279" in q_adjacency["claims"]:
                    classes |= utils.get_relation_objects(q_adjacency["claims"]["(-)P279"]) & reached_q_w_decisions

                for cl in classes:
                    if cl in compare and cl in q_gold_kept:
                        q_gold_kept_seen.add(cl)
                    elif cl in compare and cl in q_gold_pruned:
                        q_gold_pruned_seen.add(cl)
                    elif cl not in compare and cl in q_gold_kept:
                        q_gold_kept_unseen.add(cl)
                    elif cl not in compare and cl in q_gold_pruned:
                        q_gold_pruned_unseen.add(cl)
                    
                while classes:
                    seen_classes |= classes
                    new_classes = set()

                    for cl in classes:
                                
                        cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)
                        if cl_adjacency is not None:
                            sub_classes = utils.get_unseen_subclasses(cl_adjacency, seen_classes) & reached_q_w_decisions
                            for sub in sub_classes:
                                if sub in compare and sub in q_gold_kept:
                                    q_gold_kept_seen.add(sub)
                                elif sub in compare and sub in q_gold_pruned:
                                    q_gold_pruned_seen.add(sub)
                                elif sub in q_gold_kept:
                                    q_gold_kept_unseen.add(sub)
                                elif sub in q_gold_pruned:
                                    q_gold_pruned_unseen.add(sub)
                            new_classes |= sub_classes

                    classes = new_classes

            q_classifier_pruned = {
                q2 for q2 in classifier_decisions[fold][q] if classifier_decisions[fold][q][q2]["decision"] == 0
            }
            q_classifier_kept = {
                q2 for q2 in classifier_decisions[fold][q] if classifier_decisions[fold][q][q2]["decision"] == 1
            }

            reached_q_w_decisions = q_classifier_kept | q_classifier_pruned

            if q_adjacency is not None and ("P31" in q_adjacency["claims"] or "P279" in q_adjacency["claims"] or
                                "(-)P279" in q_adjacency["claims"]):
                    
                classes = set()

                if "P31" in q_adjacency["claims"]:
                    classes |= utils.get_relation_objects(q_adjacency["claims"]["P31"]) & reached_q_w_decisions
                if "P279" in q_adjacency["claims"]:
                    classes |= utils.get_relation_objects(q_adjacency["claims"]["P279"]) & reached_q_w_decisions
                if "(-)P279" in q_adjacency["claims"]:
                    classes |= utils.get_relation_objects(q_adjacency["claims"]["(-)P279"]) & reached_q_w_decisions

                for cl in classes:
                    if cl in compare:
                        seen_cl.add(cl)
                    else:
                        unseen_cl.add(cl)
                seen_classes = {q}
                while classes:
                    seen_classes |= classes
                    new_classes = set()

                    for cl in classes:
                                
                        cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)
                        if cl_adjacency is not None:
                            sub_classes = utils.get_unseen_subclasses(cl_adjacency, seen_classes) & reached_q_w_decisions
                            for sub in sub_classes:
                                if sub in compare:
                                    seen_cl.add(sub)
                                else:
                                    unseen_cl.add(sub)
                            new_classes |= sub_classes

                    classes = new_classes

            fold_precision_num += len(q_classifier_kept & q_gold_kept)
            fold_precision_den += len(q_classifier_kept)
            fold_recall_num += len(q_classifier_kept & q_gold_kept)
            fold_recall_den += len(q_gold_kept)
            fold_acc_num += len(q_classifier_kept & q_gold_kept) + len(q_classifier_pruned & q_gold_pruned)
            fold_acc_den += len(q_gold_kept) + len(q_gold_pruned)

            fold_precision_num_unseen += len(q_classifier_kept.intersection(unseen_cl) & q_gold_kept_unseen)
            fold_precision_den_unseen += len(q_classifier_kept.intersection(unseen_cl))
            fold_recall_num_unseen += len(q_classifier_kept.intersection(unseen_cl) & q_gold_kept_unseen)
            fold_recall_den_unseen += len(q_gold_kept_unseen)
            fold_acc_num_unseen += len(q_classifier_kept.intersection(unseen_cl) & q_gold_kept_unseen) + len(q_classifier_pruned.intersection(unseen_cl) & q_gold_pruned_unseen)
            fold_acc_den_unseen += len(q_gold_kept_unseen) + len(q_gold_pruned_unseen)
            
            unseen_counter += len(q_gold_kept_unseen) + len(q_gold_pruned_unseen)

            fold_precision_num_seen += len(q_classifier_kept.intersection(seen_cl) & q_gold_kept_seen)
            fold_precision_den_seen += len(q_classifier_kept.intersection(seen_cl))
            fold_recall_num_seen += len(q_classifier_kept.intersection(seen_cl) & q_gold_kept_seen)
            fold_recall_den_seen += len(q_gold_kept_seen)
            fold_acc_num_seen += len(q_classifier_kept.intersection(seen_cl) & q_gold_kept_seen) + len(q_classifier_pruned.intersection(seen_cl) & q_gold_pruned_seen)
            fold_acc_den_seen += len(q_gold_kept_seen) + len(q_gold_pruned_seen)

            seen_counter += len(q_gold_kept_seen) + len(q_gold_pruned_seen)

        my_stats[fold] = dict()
        my_stats[fold]["seen"] = seen_counter
        my_stats[fold]["unseen"] = unseen_counter

        precisions[fold] = float(fold_precision_num) / (float(fold_precision_den))
        recalls[fold] = float(fold_recall_num) / (float(fold_recall_den))
        f1[fold] = 2 * (float(precisions[fold]) * float(recalls[fold])) \
                        / (float(precisions[fold]) + float(recalls[fold]))
        acc[fold] = float(fold_acc_num) / float(fold_acc_den)

        precisions_unseen[fold] = float(fold_precision_num_unseen) / (float(fold_precision_den_unseen))
        recalls_unseen[fold] = float(fold_recall_num_unseen) / (float(fold_recall_den_unseen))
        f1_unseen[fold] = 2 * (float(precisions_unseen[fold]) * float(recalls_unseen[fold])) \
                        / (float(precisions_unseen[fold]) + float(recalls_unseen[fold]))
        acc_unseen[fold] = float(fold_acc_num_unseen) / float(fold_acc_den_unseen)

        precisions_seen[fold] = float(fold_precision_num_seen) / (float(fold_precision_den_seen))
        recalls_seen[fold] = float(fold_recall_num_seen) / (float(fold_recall_den_seen))
        f1_seen[fold] = 2 * (float(precisions_seen[fold]) * float(recalls_seen[fold])) \
                        / (float(precisions_seen[fold]) + float(recalls_seen[fold]))
        acc_seen[fold] = float(fold_acc_num_seen) / float(fold_acc_den_seen)

    stats = pandas.DataFrame({
        "avg precision": [numpy.mean([precisions[fold] for fold in precisions])],
        "std precision": [numpy.std([precisions[fold] for fold in precisions])],
        "avg recall": [numpy.mean([recalls[fold] for fold in recalls])],
        "std recall": [numpy.std([recalls[fold] for fold in recalls])],
        "avg f1": [numpy.mean([f1[fold] for fold in f1])],
        "std f1": [numpy.std([f1[fold] for fold in f1])],
        "avg accuracy": [numpy.mean([acc[fold] for fold in acc])],
        "std accuracy": [numpy.std([acc[fold] for fold in acc])],
        "unseen avg precision": [numpy.mean([precisions_unseen[fold] for fold in precisions_unseen])],
        "unseen std precision": [numpy.std([precisions_unseen[fold] for fold in precisions_unseen])],
        "unseen avg recall": [numpy.mean([recalls_unseen[fold] for fold in recalls_unseen])],
        "unseen std recall": [numpy.std([recalls_unseen[fold] for fold in recalls_unseen])],
        "unseen avg f1": [numpy.mean([f1_unseen[fold] for fold in f1_unseen])],
        "unseen std f1": [numpy.std([f1_unseen[fold] for fold in f1_unseen])],
        "unseen avg accuracy": [numpy.mean([acc_unseen[fold] for fold in acc_unseen])],
        "unseen std accuracy": [numpy.std([acc_unseen[fold] for fold in acc_unseen])],
        "seen avg precision": [numpy.mean([precisions_seen[fold] for fold in precisions_seen])],
        "seen std precision": [numpy.std([precisions_seen[fold] for fold in precisions_seen])],
        "seen avg recall": [numpy.mean([recalls_seen[fold] for fold in recalls_seen])],
        "seen std recall": [numpy.std([recalls_seen[fold] for fold in recalls_seen])],
        "seen avg f1": [numpy.mean([f1_seen[fold] for fold in f1_seen])],
        "seen std f1": [numpy.std([f1_seen[fold] for fold in f1_seen])],
        "seen avg accuracy": [numpy.mean([acc_seen[fold] for fold in acc_seen])],
        "seen std accuracy": [numpy.std([acc_seen[fold] for fold in acc_seen])]
        })
    
    print(my_stats)

    for key in my_stats:
        print(f"Fold {key}: ", (my_stats[key]["seen"]/(my_stats[key]["seen"] + my_stats[key]["unseen"]))*100)
    
    stats.to_csv(args.output, header=True, index=False)

    logger.info(f"Execution time = {utils.convert(time.time() - start)}")


if __name__ == '__main__':
    main()
