"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import argparse
import csv
import logging
import time

import numpy
import pandas
import lmdb
from sklearn.linear_model import SGDClassifier

import utils
from TqdmLoggingHandler import *


def main():
    start = time.time()

    parser = argparse.ArgumentParser(prog="svm_pruning", description="Linear Support Vector Machine Classifier")
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--concatenation", dest="concatenation", help="Type of embedding concatenation",
                        choices=["translation", "horizontal"], required=True)
    parser.add_argument("--seed-qids", dest="seed_qids", help="CSV file containing QIDs of interest (one QID per line)", required=True)
    parser.add_argument("--output-statistics", dest="output_statistics", help="Output file to save statistics", required=True)
    args = parser.parse_args()

    # Logging parameters
    logger = logging.getLogger()
    tqdm_logging_handler = TqdmLoggingHandler()
    tqdm_logging_handler.setFormatter(logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(message)s"))
    logger.addHandler(tqdm_logging_handler)
    logger.setLevel(logging.INFO)

    # Load Wikidata hashmap (QID -> properties)
    dump = lmdb.open(args.wikidata_hashmap, readonly=True, readahead=False)
    wikidata_hashmap = dump.begin()

    # Load embeddings (QID URL -> embedding)
    embeddings = lmdb.open(args.embeddings_hashmap, readonly=True, readahead=False)
    embedding_hashmap = embeddings.begin()

    # Loading decision file
    decisions = pandas.read_csv(args.decisions)

    # Load QIDs of interest
    qids_of_interest = set()
    csv_file = open(args.seed_qids, 'r')
    for line in csv_file:
        qid = line.split('\n')[0]
        qids_of_interest.add(qid)
    csv_file.close()
    qids_of_interest = qids_of_interest

    stats_seen_classes = set()

    train_features = []
    train_labels = []

    logger.info("Building training data")
    for qid in tqdm.tqdm(set(decisions["from"])):
        qid_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + qid + ">", embedding_hashmap)

        if qid_emb is not None:
            for _, r in decisions[(decisions["from"] == qid)].iterrows():
                qid2_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + r["QID"] + ">", embedding_hashmap)

                if qid2_emb is not None:
                    train_labels.append(r["target"])  # 1 : Keep, 0 : Prune

                    if args.concatenation == "translation":
                        train_features.append(qid2_emb - qid_emb)
                    else:
                        train_features.append(numpy.hstack((qid_emb, qid2_emb)))

    train_features = numpy.array(train_features)
    train_labels = numpy.array(train_labels)

    weight_for_0 = 1 / numpy.count_nonzero(train_labels == 0) * (len(train_labels) / 2.0)
    weight_for_1 = 1 / numpy.count_nonzero(train_labels == 1) * (len(train_labels) / 2.0)

    logger.info("Training SVM")
    sgd_clf = SGDClassifier(random_state=42,
                            shuffle=True,
                            class_weight= {0: weight_for_0,
                                        1: weight_for_1})

    sgd_clf.fit(train_features, train_labels)

    # Test model
    logger.info("Testing SVM")
    test_set = qids_of_interest

    for q in tqdm.tqdm(test_set, desc="test"):
        q_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + q + ">", embedding_hashmap)
        q_adjacency = utils.get_hashmap_content(q, wikidata_hashmap)
        seen_classes = {q}
        stats_seen_classes |= {q}

        if q_adjacency is not None and ("P31" in q_adjacency["claims"] or "P279" in q_adjacency["claims"] or
                                    "(-)P279" in q_adjacency["claims"]):
            
            classes = set()

            if "P31" in q_adjacency["claims"]:
                classes |= utils.get_relation_objects(q_adjacency["claims"]["P31"])
            if "P279" in q_adjacency["claims"]:
                classes |= utils.get_relation_objects(q_adjacency["claims"]["P279"])
            if "(-)P279" in q_adjacency["claims"]:
                classes |= utils.get_relation_objects(q_adjacency["claims"]["(-)P279"])

            # Start expansion
            while classes:
                seen_classes |= classes
                new_classes = set()

                # Prune or keep classes to traverse
                for cl in classes:
                    cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)
                    cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)

                    if cl_emb is not None:
                        cl_feature = None
                        if args.concatenation == "translation":
                            cl_feature = cl_emb - q_emb
                        else:
                            cl_feature = numpy.hstack((q_emb, cl_emb))

                        if sgd_clf.predict([cl_feature])[0] >= 0.5:
                            stats_seen_classes.add(cl)
                            if cl_adjacency is not None:
                                new_classes |= utils.get_unseen_subclasses(cl_adjacency, seen_classes)

                classes = new_classes

    # Statistics output
    print(f"# nodes in the output KG = {len(stats_seen_classes)}")
    with open(args.output_statistics, "w") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["# nodes in the output KG", len(stats_seen_classes)])

    logger.info(f"Execution time = {utils.convert(time.time() - start)}")


if __name__ == '__main__':
    main()
