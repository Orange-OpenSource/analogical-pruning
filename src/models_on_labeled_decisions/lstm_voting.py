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
import tqdm
import tensorflow as tf

import utils
from TqdmLoggingHandler import *


def pad_lstm_sequence(seq, max_length, pad_mode="after"):
    """ zero padding after QID and classes by default """
    sequence = seq
    pad = max_length - len(seq)
    if pad >= 0:
        if pad_mode == "after":
            sequence = sequence + pad * [numpy.zeros(200)]
        elif pad_mode == "before":
            sequence = pad * [numpy.zeros(200)] + sequence
        else:
            sequence = [sequence[0]] + pad * [numpy.zeros(200)] + sequence[1:]
    else:
        sequence = [sequence[0]] + sequence[-pad+1:]
        
    return sequence


def main():

    parser = argparse.ArgumentParser(prog="lstm_pruning", description="Long-Short-Term-Memory Classifier")
    parser.add_argument("--folds", dest="folds_path", help="File containing folds", required=True)
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--predictions", help="File containing predictions", required=True)
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--output", dest="output", help="Output pickle hashmap", required=True)
    parser.add_argument("--voting-threshold", dest="voting_threshold", help="Float value representing the threshold for deciding whether prediction analogies are 1 or 0",
                            required=False, default=0.5, type=float)
    args = parser.parse_args()

    start = time.time()

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
        logger.info("Testing LSTM")
        test_set = folds[fold]["test"]
        output_decisions[fold] = dict()

        for qid in tqdm.tqdm(test_set, desc="test"):
            depth = 0
            qid_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + qid + ">", embedding_hashmap)
            output_decisions[fold][qid] = dict()

            if qid_emb is not None:

                sequences = {}

                q_adjacency = utils.get_hashmap_content(qid, wikidata_hashmap)

                seen_classes = {qid}

                if q_adjacency is not None and ("P31" in q_adjacency["claims"] or "P279" in q_adjacency["claims"] or
                                        "(-)P279" in q_adjacency["claims"]):

                    reached_q_w_decisions = set(decisions[(decisions["from"] == qid)]["QID"])

                    classes = set()

                    if "P31" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["P31"]) & reached_q_w_decisions
                    if "P279" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["P279"]) & reached_q_w_decisions
                    if "(-)P279" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["(-)P279"]) & reached_q_w_decisions

                    depth += 1

                    classes_to_explore = set()
                    seen_classes |= classes

                    for cl in classes:
                        cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)
                        if cl_emb is not None:
                            sequence = [qid_emb, cl_emb]
                            sequences[cl] = sequence
                            output_decisions[fold][qid][cl] = dict()
                            output_decisions[fold][qid][cl]["depth"] = depth
                            if predictions[fold][qid][cl] >= args.voting_threshold:
                                output_decisions[fold][qid][cl]["decision"] = 1
                                classes_to_explore.add(cl)
                            else:
                                output_decisions[fold][qid][cl]["decision"] = 0
                    # Start expansion
                    while classes_to_explore:
                        seen_classes |= classes_to_explore
                        new_classes = set()
                        depth += 1

                        # Prune or keep classes to traverse
                        for cl in classes_to_explore:
                            cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)
                            cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)

                            if cl_emb is not None:
                                if cl_adjacency is not None:
                                    sub_classes = utils.get_unseen_subclasses(cl_adjacency, seen_classes) & reached_q_w_decisions
                                    for sub_cl in sub_classes:
                                        sub_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + sub_cl + '>', embedding_hashmap)
                                        if sub_emb is not None:
                                            sequence = sequences[cl] + [sub_emb]
                                            sequences[sub_cl] = sequence
                                            output_decisions[fold][qid][sub_cl] = dict()
                                            output_decisions[fold][qid][sub_cl]["depth"] = depth
                                            if predictions[fold][qid][sub_cl] >= args.voting_threshold:
                                                output_decisions[fold][qid][sub_cl]["decision"] = 1
                                                new_classes.add(sub_cl)
                                            else:
                                                output_decisions[fold][qid][sub_cl]["decision"] = 0

                        classes_to_explore = new_classes

    pickle.dump(output_decisions, open(args.output, "wb"))
    logger.info(f"Execution time = {utils.convert(time.time() - start)}")


if __name__ == '__main__':
    main()
