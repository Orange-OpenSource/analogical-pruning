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

import lmdb
import pandas

import utils
from TqdmLoggingHandler import *


def main():
    start = time.time()

    parser = argparse.ArgumentParser(prog="depth_pruning", description="Expand and prune QIDs along the ontology "
                                                                       "hierarchy of Wikidata from a set of seed "
                                                                       "QIDs based on exploration depth -- only down")
    parser.add_argument("--depth-threshold", dest="depth_threshold", help="Depth threshold (max depth reached)",
                        required=True, type=int)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--folds", dest="folds_path", help="File containing folds", required=True)
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

    # Loading decision file
    decisions = pandas.read_csv(args.decisions)
    output_decisions = dict()

    for fold in tqdm.tqdm(folds, desc="fold"):
        test_set = folds[fold]["test"]
        output_decisions[fold] = dict()

        for q in tqdm.tqdm(test_set, desc="test"):
            reached_q_w_decisions = set(decisions[(decisions["from"] == q)]["QID"])
            q_adjacency = utils.get_hashmap_content(q, wikidata_hashmap)
            seen_classes = {q}
            depth = 1

            output_decisions[fold][q] = dict()

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
                while classes and depth <= args.depth_threshold + 1:
                    seen_classes |= classes
                    new_classes = set()

                    # Prune or keep classes to traverse
                    for cl in classes:
                        cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)

                        output_decisions[fold][q][cl] = dict()
                        output_decisions[fold][q][cl]["depth"] = depth

                        if depth <= args.depth_threshold:
                            output_decisions[fold][q][cl]["decision"] = 1
                        else:
                            output_decisions[fold][q][cl]["decision"] = 0

                        if cl_adjacency is not None and depth <= args.depth_threshold:
                            new_classes |= (utils.get_unseen_subclasses(cl_adjacency, seen_classes) &
                                            reached_q_w_decisions)

                    depth += 1
                    classes = new_classes

    pickle.dump(output_decisions, open(args.output, "wb"))
    logger.info(f"Execution time = {utils.convert(time.time() - start)}")


if __name__ == '__main__':
    main()
