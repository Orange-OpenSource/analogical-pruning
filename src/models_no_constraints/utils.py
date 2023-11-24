"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import pickle
import time

import numpy

WIKIDATA_PREFIX = "<http://www.wikidata.org/entity/"


def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def get_hashmap_content(key, hashmap):
    obj = hashmap.get(key.encode("ascii"))
    content = None
    if obj:
        content = pickle.loads(obj)
    return content


def get_hashmap_distances(key, train_decisions, embedding_hashmap):
    emb = get_hashmap_content(WIKIDATA_PREFIX + key + ">", embedding_hashmap)
    distances = None
    train_decisions_with_distances = train_decisions.copy()
    distances = []
    for _, row in train_decisions.iterrows():
        seed_emb = get_hashmap_content(WIKIDATA_PREFIX + row["from"] + ">", embedding_hashmap)
        distances.append(euclidean_distance(seed_emb, emb))
    train_decisions_with_distances["distances"] = distances
    train_decisions_with_distances = train_decisions_with_distances.sort_values(by=["distances"])
    return train_decisions_with_distances.values.tolist()


def get_relation_objects(rel_objects):
    """ Return the set of target objects for a relation """
    targets = set()
    for obj in rel_objects:
        targets.add(obj["value"])
    return targets


def get_node_degree(entity):
    """ Compute the degree of the node obj """
    node_degree = 0
    if entity is not None and "claims" in entity:
        if "P279" in entity["claims"]:
            node_degree += len(get_relation_objects(entity["claims"]["P279"]))
        if "(-)P279" in entity["claims"]:
            node_degree += len(get_relation_objects(entity["claims"]["(-)P279"]))
        if "P31" in entity["claims"]:
            node_degree += len(get_relation_objects(entity["claims"]["P31"]))
        if "(-)P31" in entity["claims"]:
            node_degree += len(get_relation_objects(entity["claims"]["(-)P31"]))
    return node_degree


def euclidean_distance(v1, v2):
    """ Compute the Euclidean distance """
    return numpy.sqrt(sum(pow(v1 - v2, 2)))


def get_unseen_subclasses(q_adjacency, seen_classes):
    subclasses = set()

    if q_adjacency is not None and "claims" in q_adjacency and "(-)P279" in q_adjacency["claims"]:
        subclasses = get_relation_objects(q_adjacency["claims"]["(-)P279"])
        subclasses -= seen_classes

    return subclasses


def get_label(txn, qid):
    """ Get english label of an entity """
    obj = txn.get(qid.encode("ascii"))

    label = ""
    if obj:
        entity = pickle.loads(obj)

        if 'labels' in entity:
            if 'en' in entity['labels']:
                label = entity['labels']['en']
            elif 'fr' in entity['labels']:
                label = entity['labels']['fr']

    if isinstance(label, list):
        label = label[0]

    label = label.replace(",", " ")
    label = label.replace(";", " ")

    return label

def pad_sequence(seq, max_length, pad_mode="after"):
    """ zero padding for sequence """
    sequence = seq
    pad = max_length - len(seq)
    if pad >= 0:
        if pad_mode == "after":
            sequence = sequence + pad * [numpy.zeros((200, 1))]  #[numpy.zeros((200, 1))] [sequence[-1]]
        elif pad_mode == "before":
            sequence = pad * [numpy.zeros((200, 1))] + sequence
        else:
            sequence = [sequence[0]] + pad * [numpy.zeros((200, 1))] + sequence[1:]
    else:
        sequence = [sequence[0]] + sequence[-pad+1:]
    return sequence                                        


def add_label(analogy_pattern, labels, valid):
    if analogy_pattern in valid:
        labels.append(1)
    else:
        labels.append(0)


def add_analogy(mode, vec_A, vec_B, vec_C, vec_D, properties, valid_analogies_pattern, analogy_pattern, features, labels):
    if analogy_pattern in ["kk", "pp"]:
        features.append([vec_A, vec_B, vec_C, vec_D])
        add_label(analogy_pattern, labels, valid_analogies_pattern)

        if "symmetry" in properties:
            features.append([vec_C, vec_D, vec_A, vec_B])
            add_label(analogy_pattern, labels, valid_analogies_pattern)

        if analogy_pattern in valid_analogies_pattern and mode == "train" and "reflexivity" in properties:
            features.extend([[vec_A, vec_B, vec_A, vec_B], [vec_C, vec_D, vec_C, vec_D]])
            add_label(analogy_pattern, labels, valid_analogies_pattern)
            add_label(analogy_pattern, labels, valid_analogies_pattern)

        if "inner-symmetry" in properties:
            features.append([vec_B, vec_A, vec_D, vec_C])
            add_label(analogy_pattern, labels, valid_analogies_pattern)
            if "symmetry" in properties:
                features.append([vec_D, vec_C, vec_B, vec_A])
                add_label(analogy_pattern, labels, valid_analogies_pattern)
            if analogy_pattern in valid_analogies_pattern and mode == "train" and "relexivity" in properties:
                features.extend([[vec_B, vec_A, vec_B, vec_A], [vec_D, vec_C, vec_D, vec_C]])
                add_label(analogy_pattern, labels, valid_analogies_pattern)
                add_label(analogy_pattern, labels, valid_analogies_pattern)

    if analogy_pattern in ["kp", "pk"]:
        features.append([vec_A, vec_B, vec_C, vec_D])
        add_label(analogy_pattern, labels, valid_analogies_pattern)

        if "symmetry" in properties:
            features.append([vec_C, vec_D, vec_A, vec_B])
            add_label(analogy_pattern, labels, valid_analogies_pattern)

        if "inner-symmetry" in properties:
            features.append([vec_B, vec_A, vec_D, vec_C])
            add_label(analogy_pattern, labels, valid_analogies_pattern)
            if "symmetry" in properties:
                features.append([vec_D, vec_C, vec_B, vec_A])
                add_label(analogy_pattern, labels, valid_analogies_pattern)


def add_sequence_analogy(mode, vec_AB, vec_CD, properties, valid_analogies_pattern, analogy_pattern, features, labels):
    if analogy_pattern in ["kk", "pp"]:
        features.append(vec_AB + vec_CD)
        add_label(analogy_pattern, labels, valid_analogies_pattern)

        if "symmetry" in properties:
            features.append(vec_CD + vec_AB)
            add_label(analogy_pattern, labels, valid_analogies_pattern)
        if analogy_pattern in valid_analogies_pattern and mode == "train" and "reflexivity" in properties:
            features.extend([vec_AB + vec_AB, vec_CD + vec_CD])
            add_label(analogy_pattern, labels, valid_analogies_pattern)
            add_label(analogy_pattern, labels, valid_analogies_pattern)

        if "inner-symmetry" in properties:
            vec_AB.reverse()
            vec_CD.reverse()
            features.append(vec_AB + vec_CD)
            add_label(analogy_pattern, labels, valid_analogies_pattern)
            if "symmetry" in properties:
                features.append(vec_CD + vec_AB)
                add_label(analogy_pattern, labels, valid_analogies_pattern)
            if analogy_pattern in valid_analogies_pattern and mode == "train" and "relexivity" in properties:
                features.extend([vec_AB + vec_AB, vec_CD + vec_CD])
                add_label(analogy_pattern, labels, valid_analogies_pattern)
                add_label(analogy_pattern, labels, valid_analogies_pattern)
            vec_AB.reverse()
            vec_CD.reverse()

    if analogy_pattern in ["kp", "pk"]:
        features.append(vec_AB + vec_CD)
        add_label(analogy_pattern, labels, valid_analogies_pattern)

        if "symmetry" in properties:
            features.append(vec_CD + vec_AB)
            add_label(analogy_pattern, labels, valid_analogies_pattern)
            
        if "inner-symmetry" in properties:
            vec_AB.reverse()
            vec_CD.reverse()
            features.append(vec_AB + vec_CD)
            add_label(analogy_pattern, labels, valid_analogies_pattern)
            if "symmetry" in properties:
                features.append(vec_CD + vec_AB)
                add_label(analogy_pattern, labels, valid_analogies_pattern)
            vec_AB.reverse()
            vec_CD.reverse()

def add_evaluation_sequence_analogy(test_set, 
                           vec_AB, 
                           vec_CD, 
                           properties):
    
    test_set.append(vec_AB + vec_CD)
    if "symmetry" in properties:
        test_set.append(vec_CD + vec_AB)

    if "inner-symmetry" in properties:
        vec_AB.reverse()
        vec_CD.reverse()
        test_set.append(vec_AB + vec_CD)
        if "symmetry" in properties:
            test_set.append(vec_CD + vec_AB)
        vec_AB.reverse()
        vec_CD.reverse()


def add_evaluation_analogy(test_set, 
                           vec_A,
                           vec_B, 
                           vec_C,
                           vec_D, 
                           properties):
    
    test_set.append([vec_A, vec_B, vec_C, vec_D])
    if "symmetry" in properties:
        test_set.append([vec_C, vec_D, vec_A, vec_B])

    if "inner-symmetry" in properties:
        test_set.append([vec_B, vec_A, vec_D, vec_C])
        if "symmetry" in properties:
            test_set.append([vec_D, vec_C, vec_B, vec_A])
