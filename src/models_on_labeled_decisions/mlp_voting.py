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

import lmdb
import numpy
import pandas
import tensorflow as tf

import utils
from TqdmLoggingHandler import *


def main():
    start = time.time()

    parser = argparse.ArgumentParser(prog="mlp_pruning", description="Multi-layer Perceptron classifier")
    parser.add_argument("--folds", dest="folds_path", help="File containing folds", required=True)
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--predictions", help="File containing predictions", required=True)
    parser.add_argument("--output", dest="output", help="Output pickle hashmap", required=True)
    parser.add_argument("--voting-threshold", dest="voting_threshold", help="Float value representing the threshold for deciding whether prediction analogies are 1 or 0",
                            required=False, default=0.5, type=float)
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
    
    # Loading predictions hashmap
    predictions = pickle.load(open(args.predictions, "rb"))

    for fold in tqdm.tqdm(folds, desc="fold"):
        # Test model
        logger.info("Testing MLP")
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
                                output_decisions[fold][q][cl] = dict()
                                output_decisions[fold][q][cl]["depth"] = depth
                                if predictions[fold][q][cl] >= args.voting_threshold:
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
