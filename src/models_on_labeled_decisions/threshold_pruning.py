"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import argparse

import lmdb
import pandas
import tqdm

from utils import *


def get_direct_classes_mean_distance(direct_classes, starting_emb, embedding_hashmap):
    """
    Compute mean distance of direct classes
    :param direct_classes: set of direct classes
    :param starting_emb: embedding of the starting QID
    :param embedding_hashmap: the embedding hashmap
    :return: the mean distance of direct classes
    """

    dist_direct = []

    for cl in direct_classes:
        cl_emb_pkl = embedding_hashmap.get((WIKIDATA_PREFIX + cl + '>').encode('ascii'))
        if cl_emb_pkl:
            cl_emb = pickle.loads(cl_emb_pkl)

            if len(cl_emb) != 0:
                dist_direct.append(euclidean_distance(starting_emb, cl_emb))

    mean_direct_classes = 0
    if len(dist_direct) != 0:
        q3, q1 = numpy.percentile(dist_direct, [75, 25])
        iqr = q3 - q1
        dist_outliers = q3 + 1.5 * iqr
        nb_cl = 0

        for dist_d in dist_direct:
            # Mean distance of direct classes is computed excluding outliers
            if dist_d <= dist_outliers:
                nb_cl += 1
                mean_direct_classes += dist_d

        mean_direct_classes = mean_direct_classes / nb_cl

    return mean_direct_classes


def main():
    start = time.time()

    parser = argparse.ArgumentParser(prog="threshold_pruning", description="Expand and prune QIDs along the ontology "
                                                                           "hierarchy of Wikidata from a set of seed "
                                                                           "QIDs based on node degree and embedding "
                                                                           "distance thresholds -- only down")
    parser.add_argument("--nd-threshold", dest="node_degree_abs_threshold", help="Absolute threshold for node degree",
                        required=True, type=int)
    parser.add_argument("--alpha", dest="node_degree_alpha",
                        help="Alpha coefficient for node degree relative threshold", required=True, type=float)
    parser.add_argument("--gamma", dest="node_degree_gamma", help="Gamma coefficient for node degree threshold",
                        required=True, default=20, type=int)
    parser.add_argument("--beta", dest="distance_beta", help="Beta coefficient for distance threshold", required=True,
                        type=float)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--folds", dest="folds_path", help="File containing folds", required=True)
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--output", dest="output", help="Output pickle hashmap", required=True)
    args = parser.parse_args()

    # Pruning thresholds
    node_degree_abs_threshold = args.node_degree_abs_threshold
    node_degree_alpha = args.node_degree_alpha
    node_degree_gamma = args.node_degree_gamma
    distance_beta = args.distance_beta

    # Load Wikidata hashmap (QID -> properties)
    dump = lmdb.open(args.wikidata_hashmap, readonly=True, readahead=False)
    wikidata_hashmap = dump.begin()

    # Load embeddings (QID URL -> embedding)
    embeddings = lmdb.open(args.embeddings_hashmap, readonly=True, readahead=False)
    embedding_hashmap = embeddings.begin()

    # Load folds
    folds = pickle.load(open(args.folds_path, "rb"))

    # Loading decision file
    decisions = pandas.read_csv(args.decisions)
    output_decisions = dict()

    for fold in tqdm.tqdm(folds, desc="fold"):
        test_set = folds[fold]["test"]
        output_decisions[fold] = dict()

        # Starting expansion QID by QID
        for q in tqdm.tqdm(test_set, desc="test"):
            reached_q_w_decisions = set(decisions[(decisions["from"] == q)]["QID"])

            output_decisions[fold][q] = dict()

            seen_classes = {
                "kept": set(),
                "kept but > rel deg": set(),
                "kept but > abs deg": set(),
                "pruned > rel dist": set(),
                "pruned parent > rel deg": set(),
                "pruned parent > abs deg": set()
            }

            # Get QID adjacency and embedding
            q_adjacency = get_hashmap_content(q, wikidata_hashmap)
            q_embedding = get_hashmap_content(WIKIDATA_PREFIX + q + ">", embedding_hashmap)

            # If QID adjacency exists and has embeddings and has direct classes
            if q_adjacency is not None and q_embedding is not None and ("P31" in q_adjacency["claims"] or
                                                                        "P279" in q_adjacency["claims"] or
                                                                        "(-)P279" in q_adjacency["claims"]):
                direct_classes = set()
                if "P31" in q_adjacency["claims"]:
                    direct_classes |= get_relation_objects(q_adjacency["claims"]["P31"])
                if "P279" in q_adjacency["claims"]:
                    direct_classes |= get_relation_objects(q_adjacency["claims"]["P279"])
                if "(-)P279" in q_adjacency["claims"]:
                    direct_classes |= get_relation_objects(q_adjacency["claims"]["(-)P279"])

                direct_classes_mean_distance = get_direct_classes_mean_distance(direct_classes, q_embedding,
                                                                                embedding_hashmap)

                # Start expansion
                classes_to_traverse = direct_classes & reached_q_w_decisions
                depth = 1
                while classes_to_traverse:
                    new_classes = set()

                    # Compute degrees and relative threshold
                    degrees = []
                    for cl in classes_to_traverse:
                        degrees.append(get_node_degree(cl))
                    q3, q1 = numpy.percentile(degrees, [75, 25])
                    max_degree = numpy.max(degrees)
                    relative_degree_threshold = q3 + node_degree_alpha * (q3 - q1)

                    # Prune or keep classes to traverse
                    for cl in classes_to_traverse:
                        cl_adjacency = get_hashmap_content(cl, wikidata_hashmap)
                        cl_emb = get_hashmap_content(WIKIDATA_PREFIX + cl + '>', embedding_hashmap)

                        cl_degree = get_node_degree(cl_adjacency)

                        if cl_emb is not None:
                            if cl not in output_decisions[fold][q]:
                                output_decisions[fold][q][cl] = dict()

                            # Relative distance pruning
                            if euclidean_distance(q_embedding, cl_emb) > distance_beta * direct_classes_mean_distance:
                                seen_classes["pruned > rel dist"].add(cl)
                                output_decisions[fold][q][cl]["decision"] = 0
                                output_decisions[fold][q][cl]["depth"] = depth

                            # Else
                            else:
                                # Relative degree pruning
                                if max_degree > node_degree_gamma and cl_degree > relative_degree_threshold:
                                    seen_classes["kept but > rel deg"].add(cl)
                                    output_decisions[fold][q][cl]["decision"] = 1
                                    output_decisions[fold][q][cl]["depth"] = depth

                                    for subc in reached_q_w_decisions & \
                                                get_unseen_subclasses(cl_adjacency, seen_classes["kept"] |
                                                                                    seen_classes["pruned > rel dist"] |
                                                                                    seen_classes["kept but > rel deg"] |
                                                                                    seen_classes["kept but > abs deg"]):
                                        if subc not in output_decisions[fold][q]:
                                            output_decisions[fold][q][subc] = dict()

                                        output_decisions[fold][q][subc]["decision"] = 0
                                        output_decisions[fold][q][subc]["depth"] = depth

                                        seen_classes["pruned parent > rel deg"].add(subc)

                                # Absolute degree threshold
                                elif cl_degree > node_degree_abs_threshold:
                                    seen_classes["kept but > abs deg"].add(cl)
                                    output_decisions[fold][q][cl]["decision"] = 1
                                    output_decisions[fold][q][cl]["depth"] = depth

                                    for subc in reached_q_w_decisions & \
                                                get_unseen_subclasses(cl_adjacency, seen_classes["kept"] |
                                                                                    seen_classes["pruned > rel dist"] |
                                                                                    seen_classes["kept but > rel deg"] |
                                                                                    seen_classes["kept but > abs deg"]):
                                        if subc not in output_decisions[fold][q]:
                                            output_decisions[fold][q][subc] = dict()

                                        output_decisions[fold][q][subc]["decision"] = 0
                                        output_decisions[fold][q][subc]["depth"] = depth

                                        seen_classes["pruned parent > abs deg"].add(subc)

                                # Keep
                                else:
                                    seen_classes["kept"].add(cl)
                                    seen_classes["pruned parent > rel deg"] -= {cl}
                                    seen_classes["pruned parent > abs deg"] -= {cl}
                                    output_decisions[fold][q][cl]["decision"] = 1
                                    output_decisions[fold][q][cl]["depth"] = depth

                                    new_classes |= reached_q_w_decisions & \
                                                   get_unseen_subclasses(cl_adjacency, seen_classes["kept"] |
                                                                         seen_classes["pruned > rel dist"] |
                                                                         seen_classes["kept but > rel deg"] |
                                                                         seen_classes["kept but > abs deg"])

                    classes_to_traverse = new_classes
                    depth += 1

    pickle.dump(output_decisions, open(args.output, "wb"))
    dump.close()
    print("Execution time = ", convert(time.time() - start))


if __name__ == '__main__':
    main()
