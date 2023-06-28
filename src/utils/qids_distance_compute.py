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
import tqdm

from utils import get_hashmap_content
from utils import euclidean_distance
from utils import WIKIDATA_PREFIX
from utils import convert


def main():
    start = time.time()

    parser = argparse.ArgumentParser(prog="qids_distance_compute",
                                     description="Compute the distance between each pair of QIDs of interest")
    parser.add_argument("--qids", help="CSV file containing QIDs of interest (one QID per line)", required=True)
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--lmdb-size", dest="lmdb_size", help="Size of the output lmdb hashmap", type=int,
                        required=True)
    parser.add_argument("--output", dest="output", help="Output hashmap", required=True)
    args = parser.parse_args()

    # Load QIDs of interest
    qids_of_interest = set()
    csv_file = open(args.qids, 'r')
    for line in csv_file:
        qid = line.split('\n')[0]
        qids_of_interest.add(qid)
    csv_file.close()
    qids_of_interest = list(qids_of_interest)

    # Load embeddings (QID URL -> embedding)
    embeddings = lmdb.open(args.embeddings_hashmap, readonly=True, readahead=False)
    embedding_hashmap = embeddings.begin()

    # Create LMDB hashmap
    distances = lmdb.open(args.output, map_size=int(args.lmdb_size))
    distances_hashmap = distances.begin(write=True)

    # Create dicts
    for qid in tqdm.tqdm(qids_of_interest):
        distances_hashmap.put(qid.encode('ascii'), pickle.dumps(dict()))

    # Fill dict with distances
    for i, qid1 in tqdm.tqdm(enumerate(qids_of_interest), total=len(qids_of_interest)):
        qid1_emb = get_hashmap_content(WIKIDATA_PREFIX + qid1 + ">", embedding_hashmap)

        if qid1_emb is not None:
            for j in range(i + 1, len(qids_of_interest)):
                qid2 = qids_of_interest[j]
                qid2_emb = get_hashmap_content(WIKIDATA_PREFIX + qid2 + ">", embedding_hashmap)

                if qid2_emb is not None:
                    d = euclidean_distance(qid1_emb, qid2_emb)

                    qid1_dist = pickle.loads(distances_hashmap.get(qid1.encode("ascii")))
                    qid2_dist = pickle.loads(distances_hashmap.get(qid2.encode("ascii")))

                    qid1_dist[qid2] = d
                    qid2_dist[qid1] = d

                    distances_hashmap.put(qid1.encode("ascii"), pickle.dumps(qid1_dist))
                    distances_hashmap.put(qid2.encode("ascii"), pickle.dumps(qid2_dist))

    distances_hashmap.commit()
    distances.close()
    print("Execution time = ", convert(time.time() - start))


if __name__ == '__main__':
    main()
