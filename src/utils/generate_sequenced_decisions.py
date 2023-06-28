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
import tqdm

import utils
from TqdmLoggingHandler import *


def main():

    parser = argparse.ArgumentParser(prog="generate_sequenced_decisions", 
                                     description="Generate sequenced decisions with path in the graph for sequenced analogies training")
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--output", dest="output", help="Output pickle hashmap", required=True)
    args = parser.parse_args()

    start = time.time()


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

    starting_qids = set(decisions["from"])

    sequences = {}

    for qid in tqdm.tqdm(starting_qids):

        qid_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + qid + ">", embedding_hashmap)

        if qid_emb is not None:

            qid_sequences = {}

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

                for cl in classes:
                    cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)
                    if cl_emb is not None:
                        sequence = [numpy.transpose(numpy.array([qid_emb]), (1, 0)), numpy.transpose(numpy.array([cl_emb]))]
                        qid_sequences[cl] = sequence

                # Start expansion
                while classes:
                    seen_classes |= classes
                    new_classes = set()

                    # Prune or keep classes to traverse
                    for cl in classes:
                        cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)
                        cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)

                        if cl_emb is not None and cl_adjacency is not None:
                            sub_classes = utils.get_unseen_subclasses(cl_adjacency, seen_classes) & reached_q_w_decisions
                            for sub_cl in sub_classes:
                                sub_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + sub_cl + '>', embedding_hashmap)
                                if sub_emb is not None:
                                    sequence = qid_sequences[cl] + [numpy.transpose(numpy.array([sub_emb]))]
                                    qid_sequences[sub_cl] = sequence
                            new_classes |= sub_classes

                    classes = new_classes

        sequences[qid] = qid_sequences

    pickle.dump(sequences, open(args.output, "wb"))
    logger.info(f"Execution time = {utils.convert(time.time() - start)}")


if __name__ == '__main__':
    main()
