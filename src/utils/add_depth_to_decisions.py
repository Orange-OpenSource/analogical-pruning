"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import time
import argparse
import logging

import lmdb
import pandas

import utils
from TqdmLoggingHandler import *


def main():
    start = time.time()

    parser = argparse.ArgumentParser(prog="add_depth_to_decisions", description="Add depth of the kept or pruned class in each decision for further uses")
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap", required=True)
    parser.add_argument("--output", dest="output", help="decisions file with depth column", required=True)
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

    # Loading decision file
    decisions = pandas.read_csv(args.decisions)

    starting_qids = set(decisions['from'])
    
    reached_class = []
    reached_class_label = []
    starting_entity = []
    starting_entity_label = []
    decision_target = []
    reached_class_depth = []

    for qid in starting_qids:
        
        q_adjacency = utils.get_hashmap_content(qid, wikidata_hashmap)
        seen_classes = set()
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

            depth = 0
            while classes:
                depth += 1
                seen_classes |= classes
                new_classes = set()

                for cl in classes:
                    # Add depth of reached class
                    decision = decisions[(decisions['from'] == qid) & (decisions['QID'] == cl)]
                    reached_class.append(cl)
                    reached_class_label.append(decision["label"].iloc[0])
                    starting_entity.append(qid)
                    starting_entity_label.append(decision["starting label"].iloc[0])
                    decision_target.append(decision["target"].iloc[0])
                    reached_class_depth.append(depth)

                    cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)
                    if cl_adjacency is not None:
                        sub_classes = utils.get_unseen_subclasses(cl_adjacency, seen_classes) & reached_q_w_decisions
                        new_classes |= sub_classes

                classes = new_classes

    dump.close()

    decisions_with_depth = pandas.DataFrame({"QID": reached_class, 
                                             "label": reached_class_label, 
                                             "from": starting_entity, 
                                             "starting label": starting_entity_label, 
                                             "target": decision_target, 
                                             "depth": reached_class_depth})
    
    decisions_with_depth.to_csv(args.output, columns=["from", "starting label", "QID", "label", "depth", "target"], index=False, header=True)
    
    logger.info(f"Execution time = {utils.convert(time.time() - start)}")


if __name__ == '__main__':
    main()