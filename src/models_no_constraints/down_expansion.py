"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import argparse
import csv

import pandas as pd
import lmdb
import tqdm

from utils import *


def main():
    start = time.time()
    
    parser = argparse.ArgumentParser(prog="down_expansion", description="Expand QIDs along the ontology "
                                                                        "hierarchy of Wikidata from a set of "
                                                                        "seed QIDs -- only down -- no pruning")
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--seed-qids", dest="seed_qids", help="Seed QIDs to use in a CSV file", required=True)
    parser.add_argument("--output", dest="output", help="Output file to save statistics", required=True)
    args = parser.parse_args()

    # Load Wikidata hashmap (QID -> properties)
    dump = lmdb.open(args.wikidata_hashmap, readonly=True, readahead=False)
    wikidata_hashmap = dump.begin()
    
    # Load seed QIDs
    qids_to_expand = set()
    csv_file = open(args.seed_qids, 'r')
    for line in csv_file:
        qid = line.split('\n')[0]
        qids_to_expand.add(qid)
    csv_file.close()
    
    # Read starting QIDs to expand
    seen_classes = set()

    # Starting expansion QID by QID
    for q in tqdm.tqdm(qids_to_expand):
        q_adjacency = get_hashmap_content(q, wikidata_hashmap)
        seen_classes |= {q}

        if q_adjacency is not None and ("P31" in q_adjacency["claims"] or "P279" in q_adjacency["claims"] or
                                        "(-)P279" in q_adjacency["claims"]):

            classes = set()

            if "P31" in q_adjacency["claims"]:
                classes |= get_relation_objects(q_adjacency["claims"]["P31"])
            if "P279" in q_adjacency["claims"]:
                classes |= get_relation_objects(q_adjacency["claims"]["P279"])
            if "(-)P279" in q_adjacency["claims"]:
                classes |= get_relation_objects(q_adjacency["claims"]["(-)P279"])

            # Start expansion
            while classes:
                seen_classes |= classes
                new_classes = set()

                for cl in classes:
                    cl_adjacency = get_hashmap_content(cl, wikidata_hashmap)

                    if cl_adjacency is not None and "claims" in cl_adjacency:
                        if "(-)P279" in cl_adjacency["claims"]:
                            cl_subclasses = get_relation_objects(cl_adjacency["claims"]["(-)P279"])
                            new_classes |= (cl_subclasses - seen_classes)

                classes = new_classes

    dump.close()

    print(f"# nodes in the output KG = {len(seen_classes)}")

    with open(args.output, "w") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["# nodes in the output KG", len(seen_classes)])

    print("Execution time = ", convert(time.time() - start))
    
        
if __name__ == '__main__':   
    main()
    
