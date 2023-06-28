"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import argparse
import csv
import time

import pandas as pd
import lmdb

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
    parser.add_argument("--seed-qids", dest="seed_qids", help="Seed QIDs to use in a CSV file", required=True)
    parser.add_argument("--output", dest="output", help="Output file to save statistics", required=True)
    args = parser.parse_args()

    # Logging parameters
    logger = logging.getLogger()
    tqdm_logging_handler = TqdmLoggingHandler()
    tqdm_logging_handler.setFormatter(logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(message)s"))
    logger.addHandler(tqdm_logging_handler)
    logger.setLevel(logging.INFO)

    # Load seed QIDs
    qids_to_expand = set()
    csv_file = open(args.seed_qids, 'r')
    for line in csv_file:
        qid = line.split('\n')[0]
        qids_to_expand.add(qid)
    csv_file.close()

    # Load Wikidata hashmap (QID -> properties)
    dump = lmdb.open(args.wikidata_hashmap, readonly=True, readahead=False)
    wikidata_hashmap = dump.begin()

    stats_seen_classes = set()

    for q in tqdm.tqdm(qids_to_expand):
        q_adjacency = utils.get_hashmap_content(q, wikidata_hashmap)
        seen_classes = {q}
        depth = 1

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
            while classes and depth <= args.depth_threshold:
                seen_classes |= classes
                new_classes = set()

                # Prune or keep classes to traverse
                for cl in classes:
                    cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)

                    if cl_adjacency is not None and depth < args.depth_threshold:
                        new_classes |= (utils.get_unseen_subclasses(cl_adjacency, seen_classes))

                depth += 1
                classes = new_classes

        stats_seen_classes |= seen_classes

    # Statistics output
    print(f"# nodes in the output KG = {len(stats_seen_classes)}")
    with open(args.output, "w") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["# nodes in the output KG", len(stats_seen_classes)])

    logger.info(f"Execution time = {utils.convert(time.time() - start)}")


if __name__ == '__main__':
    main()
