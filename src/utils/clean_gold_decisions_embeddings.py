"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import argparse
import time

import lmdb
import pandas

import utils
from TqdmLoggingHandler import *


def main():
    start = time.time()

    parser = argparse.ArgumentParser(prog="clean_gold_decisions", description="Avoid keeping decisions reached "
                                                                              "further in the hierarchy than  "
                                                                              "other prune decisions in the "
                                                                              "labeled dataset")
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--output", dest="output", help="Output cleaned decision file", required=True)
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

    decisions.drop_duplicates(inplace=True)

    # Get seed QIDs
    starting_qids = set(decisions['from'])
    decision_qids = set(decisions['QID'])
    starting_labels = {q: list(decisions[(decisions["from"] == q)]["starting label"])[0] for q in starting_qids}
    decision_labels = {q: list(decisions[(decisions["QID"] == q)]["label"])[0] for q in decision_qids}

    # Create output decision file
    output_decisions = []

    for q in tqdm.tqdm(starting_qids):
        qid_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + q + ">", embedding_hashmap)
        if qid_emb is not None:
            q_adjacency = utils.get_hashmap_content(q, wikidata_hashmap)
            kept_classes = set(decisions[(decisions["from"] == q) & (decisions["target"] == 1)]["QID"])
            pruned_classes = set(decisions[(decisions["from"] == q) & (decisions["target"] == 0)]["QID"])
            seen_classes = {q}
            depth = 0

            # Check for contradictory decisions
            for c in kept_classes & pruned_classes:
                logger.error(f"Contradictory decision for {q} and {c}")

            if q_adjacency is not None and ("P31" in q_adjacency["claims"] or "P279" in q_adjacency["claims"] or
                                            "(-)P279" in q_adjacency["claims"]):
                classes = set()

                if "P31" in q_adjacency["claims"]:
                    classes |= utils.get_relation_objects(q_adjacency["claims"]["P31"]) & (kept_classes | pruned_classes)
                if "P279" in q_adjacency["claims"]:
                    classes |= utils.get_relation_objects(q_adjacency["claims"]["P279"]) & (kept_classes | pruned_classes)
                if "(-)P279" in q_adjacency["claims"]:
                    classes |= utils.get_relation_objects(q_adjacency["claims"]["(-)P279"]) & (kept_classes | pruned_classes)

                # Start expansion
                while classes:
                    depth += 1
                    seen_classes |= classes
                    new_classes = set()

                    for cl in classes:
                        cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)
                        if cl_emb is not None:
                            output_decisions.append({
                                "QID": cl,
                                "label": decision_labels[cl],
                                "from": q,
                                "starting label": starting_labels[q],
                                "target": int(cl in kept_classes),
                                "depth" : depth
                            })

                            cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)

                            if cl_adjacency is not None and "claims" in cl_adjacency and cl in kept_classes:
                                if "(-)P279" in cl_adjacency["claims"]:
                                    cl_subclasses = utils.get_relation_objects(cl_adjacency["claims"]["(-)P279"])
                                    new_classes |= (cl_subclasses & (kept_classes | pruned_classes) ) - seen_classes

                    classes = new_classes

    output_decisions = pandas.DataFrame(output_decisions, columns=["QID", "label", "from", "starting label", "target", "depth"])
    output_decisions.to_csv(args.output, index=False)
    logger.info(f"Execution time = {utils.convert(time.time() - start)}")


if __name__ == '__main__':
    main()
