"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import argparse
import logging
import pickle
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
    parser.add_argument("--folds", dest="folds_path", help="File containing folds", required=True)
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--concatenation", dest="concatenation", help="Type of embedding concatenation",
                        choices=["translation", "horizontal"], required=True)
    parser.add_argument("--output", dest="output", help="Output pickle hashmap", required=True)
    args = parser.parse_args()

    # Logging parameters
    logger = logging.getLogger()
    tqdm_logging_handler = TqdmLoggingHandler()
    tqdm_logging_handler.setFormatter(logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(message)s"))
    logger.addHandler(tqdm_logging_handler)
    logger.setLevel(logging.INFO)

    # Load folds
    folds = pickle.load(open(args.folds_path, "rb"))

    # Load Wikidata hashmap (QID -> properties)
    dump = lmdb.open(args.wikidata_hashmap, readonly=True, readahead=False)
    wikidata_hashmap = dump.begin()

    # Load embeddings (QID URL -> embedding)
    embeddings = lmdb.open(args.embeddings_hashmap, readonly=True, readahead=False)
    embedding_hashmap = embeddings.begin()

    # Loading decision file
    decisions = pandas.read_csv(args.decisions)
    output_decisions = dict()

    for fold in tqdm.tqdm(folds, desc="fold"):

        val_set = set(folds[fold]["train"][:int(len(folds[fold]["train"])/5)])

        # Train model
        train_set = set(folds[fold]["train"]) - val_set

        train_features = []
        train_labels = []

        logger.info("Building training data")
        for qid in tqdm.tqdm(train_set):
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
        test_set = folds[fold]["test"]
        output_decisions[fold] = dict()

        for q in tqdm.tqdm(test_set, desc="test"):
            q_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + q + ">", embedding_hashmap)
            if q_emb is not None:
                reached_q_w_decisions = set(decisions[(decisions["from"] == q)]["QID"])
                q_adjacency = utils.get_hashmap_content(q, wikidata_hashmap)
                seen_classes = {q}
                depth = 0

                output_decisions[fold][q] = dict()

                if q_adjacency is not None and ("P31" in q_adjacency["claims"] or "P279" in q_adjacency["claims"] or
                                                "(-)P279" in q_adjacency["claims"]):

                    classes = set()
                    
                    if "P31" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["P31"]) & reached_q_w_decisions
                    if "P279" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["P279"]) & reached_q_w_decisions
                    if "(-)P279" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["(-)P279"]) & reached_q_w_decisions

                    # Start expansion
                    while classes:
                        depth += 1
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

                                output_decisions[fold][q][cl] = dict()
                                output_decisions[fold][q][cl]["depth"] = depth
                                if sgd_clf.predict([cl_feature])[0] >= 0.5:
                                    output_decisions[fold][q][cl]["decision"] = 1
                                    if cl_adjacency is not None:
                                        new_classes |= (utils.get_unseen_subclasses(cl_adjacency, seen_classes) & 
                                                        reached_q_w_decisions)
                                else:
                                    output_decisions[fold][q][cl]["decision"] = 0

                        classes = new_classes

    pickle.dump(output_decisions, open(args.output, "wb"))
    logger.info(f"Execution time = {utils.convert(time.time() - start)}")


if __name__ == '__main__':
    main()
