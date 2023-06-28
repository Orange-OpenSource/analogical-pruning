"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import pickle
import argparse

import numpy
import lmdb
import pandas

import utils
from TqdmLoggingHandler import *


def main():

    parser = argparse.ArgumentParser(prog="analogy_evaluation", description="Evaluation of analogy-based models")
    parser.add_argument("--voting", help="Voting method for expansion (evaluation of analogy model)", required=False, default="majority")
    parser.add_argument("--voting-threshold", dest="voting_threshold", help="Float value representing the threshold for deciding whether prediction analogies are 1 or 0",
                            required=False, default=0.5, type=float)
    parser.add_argument("--folds", dest="folds_path", help="File containing folds", required=True)
    parser.add_argument("--predictions", help="File containing folds", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--valid-analogies-pattern", dest="valid_analogies_pattern", nargs='+')
    parser.add_argument("--invalid-analogies-pattern", dest="invalid_analogies_pattern", nargs='+')
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
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

    # Loading predictions hashmap
    predictions = pickle.load(open(args.predictions, "rb"))

    # Loading decision file
    decisions = pandas.read_csv(args.decisions)
    output_decisions = dict()

    logger.info("Expansion --> evaluation of decisions")

    for fold in tqdm.tqdm(folds, desc="fold"):

        test_set = folds[fold]["test"]
        
        # Decisions evalution
        output_decisions[fold] = dict()
        # Start the expansion
        for qid in tqdm.tqdm(test_set):
            depth = 0
            qid_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + qid + ">", embedding_hashmap)
            output_decisions[fold][qid] = dict()

            if qid_emb is not None:

                q_adjacency = utils.get_hashmap_content(qid, wikidata_hashmap)
                seen_classes = {qid}
                reached_q_w_decisions = set(decisions[(decisions["from"] == qid)]["QID"])

                if q_adjacency is not None and ("P31" in q_adjacency["claims"] or "P279" in q_adjacency["claims"] or
                                "(-)P279" in q_adjacency["claims"]):
                    
                    classes = set()

                    if "P31" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["P31"]) & reached_q_w_decisions
                    if "P279" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["P279"]) & reached_q_w_decisions
                    if "(-)P279" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["(-)P279"]) & reached_q_w_decisions

                    while classes:
                        seen_classes |= classes
                        new_classes = set()
                        depth += 1

                        for cl in classes:
                            cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)
                            if cl_emb is not None:
                                output_decisions[fold][qid][cl] = dict()
                                output_decisions[fold][qid][cl]["depth"] = depth

                                decision = 0
                                # Analogy predictions
                                keep = 0
                                prune = 0

                                keep_counter = []
                                prune_counter = []

                                if "keeping" in predictions[fold][qid][cl]:
                                    y_keeping_pred = predictions[fold][qid][cl]["keeping"]
                                    if args.voting == "majority":
                                        keep_counter = (y_keeping_pred >= args.voting_threshold).astype('int32')
                                        keep += numpy.count_nonzero(keep_counter == 1)
                                        prune += numpy.count_nonzero(keep_counter == 0)
                                    if args.voting == "weighted":
                                        keep = y_keeping_pred

                                if "pruning" in predictions[fold][qid][cl]:
                                    y_pruning_pred = predictions[fold][qid][cl]["pruning"]
                                    predictions[fold][qid][cl]["pruning"] = y_pruning_pred
                                    if args.voting == "majority":
                                        prune_counter = (y_pruning_pred < args.voting_threshold).astype('int32')
                                        prune += numpy.count_nonzero(prune_counter == 1)
                                        keep += numpy.count_nonzero(prune_counter == 0)
                                    if args.voting == "weighted":
                                        prune = y_pruning_pred
                                # Voting
                                if args.voting == "weighted":
                                    if "kk" in args.valid_analogies_pattern and "pp" not in args.valid_analogies_pattern:
                                        keep = numpy.mean(keep)
                                    elif "pp" in args.valid_analogies_pattern and "kk" not in args.valid_analogies_pattern:
                                        if "pruning" in predictions[fold][qid][cl]:
                                            keep = numpy.mean(1-prune)
                                    elif "pp" in args.valid_analogies_pattern and "kk" in args.valid_analogies_pattern:
                                        if "pruning" in predictions[fold][qid][cl]:
                                            keep = numpy.mean(numpy.concatenate((keep, 1 - prune)))
                                        else:
                                            keep = numpy.mean(keep)
                                    if keep >= args.voting_threshold:
                                        decision = 1
                                    else:
                                        decision = 0  
                                elif args.voting == "majority":
                                    if keep >= args.voting_threshold:
                                        decision = 1
                                    else:
                                        decision = 0

                                cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)
                                if decision == 1 and cl_adjacency is not None:
                                    new_classes |= utils.get_unseen_subclasses(cl_adjacency, seen_classes) & reached_q_w_decisions
                                output_decisions[fold][qid][cl]["decision"] = decision

                        classes = new_classes

    pickle.dump(output_decisions, open(args.output, "wb"))


if __name__ == '__main__':
    main()
