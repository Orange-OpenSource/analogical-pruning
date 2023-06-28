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

from utils import convert
from utils import get_label


def main():
    start = time.time()

    parser = argparse.ArgumentParser(prog="get_labels", description="Get labels of a list of QIDs in a CSV file")
    parser.add_argument("--qids", help="CSV file containing QIDs (one QID per line)", required=True)
    parser.add_argument("--wikidata", help="Folder containing the LMDB hashmap of Wikidata", required=True)
    parser.add_argument("--output", help="Output CSV file storing on each line QID,label", required=True)
    args = parser.parse_args()

    dump = lmdb.open(args.wikidata, readonly=True, readahead=False)     
    txn = dump.begin()

    qids = []
    labels = []

    csv_file = open(args.qids, 'r')

    for line in csv_file:
        qid = line.split('\n')[0]
        label = get_label(txn, qid)

        if isinstance(label, list):
            label = label[0]

        label = label.replace(",", " ")
        label = label.replace(";", " ")

        if label != "":
            qids.append(qid)
            labels.append(label)

    csv_file.close()
    print(f"Number of remaining QIDs: {len(qids)}")

    df = pandas.DataFrame({"qid": qids, "label": labels})

    df.to_csv(args.output, index=False, header=True)
    
    end = time.time()
    print("Execution time = ", convert(end-start))


if __name__ == '__main__':   
    main()
