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


def get_hashmap_content(key, hashmap):
    obj = hashmap.get(key.encode("ascii"))
    content = None
    if obj:
        content = pickle.loads(obj)
    return content


def get_relation_objects(rel_objects):
    """ Return the set of target objects for a relation """
    targets = set()
    for obj in rel_objects:
        targets.add(obj["value"])
    return targets


def get_unseen_subclasses(q_adjacency, seen_classes):
    subclasses = set()

    if q_adjacency is not None and "claims" in q_adjacency and "(-)P279" in q_adjacency["claims"]:
        subclasses = get_relation_objects(q_adjacency["claims"]["(-)P279"])
        subclasses -= seen_classes

    return subclasses


def euclidean_distance(v1, v2):
    """ Compute the Euclidean distance """
    return numpy.sqrt(sum(pow(v1 - v2, 2)))
