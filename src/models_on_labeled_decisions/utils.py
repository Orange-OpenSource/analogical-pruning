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


def get_hashmap_distances(key, hashmap, train_decisions):
    """ """
    obj = hashmap.get(key.encode("ascii"))
    distances = None
    train_decisions_with_distances = train_decisions.copy()
    distances = []
    if obj:
        content = pickle.loads(obj)
        for ind, row in train_decisions.iterrows():
            if row["from"] in content:
                distances.append(content[row["from"]])
            else:
                train_decisions_with_distances.drop(ind, inplace=True)
    train_decisions_with_distances["distances"] = distances
    train_decisions_with_distances = train_decisions_with_distances.sort_values(by=["distances"])

    return train_decisions_with_distances.values.tolist()


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


def euclidean_distance(v1, v2):
    """ Compute the Euclidean distance """
    return numpy.sqrt(sum(pow(v1 - v2, 2)))


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
            sequence = sequence + pad * [numpy.zeros((200, 1))]  # [sequence[-1]]
        elif pad_mode == "before":
            sequence = pad * [numpy.zeros((200, 1))] + sequence
        else:
            sequence = [sequence[0]] + pad * [numpy.zeros((200, 1))] + sequence[1:]
    else:
        sequence = [sequence[0]] + sequence[-pad+1:]
    return sequence           

def pad_sequence2(seq, max_length, pad_mode="after"):
    """ zero padding for sequence """
    sequence = seq
    pad = max_length - len(seq)
    if pad >= 0:
        if pad_mode == "after":
            sequence = sequence + pad * [sequence[-1]]
        elif pad_mode == "before":
            sequence = pad * [numpy.zeros((200, 1))] + sequence
        else:
            sequence = [sequence[0]] + pad * [numpy.zeros((200, 1))] + sequence[1:]
    else:
        sequence = [sequence[0]] + sequence[-pad+1:]
    return sequence                                 


def analogies_voting(conv_model, 
                     voting,
                     voting_threshold,
                     X_keeping,
                     X_pruning,
                     valid,
                     analogy_stats,
                     target,
                     decision=[],
                     keeping_decisions=[],
                     pruning_decisions=[]):
    """ Voting of analogies to make a keeping decision or pruning decision """

    # Analogy predictions
    keep = 0
    prune = 0

    keep_counter = []
    prune_counter = []


    if len(X_keeping) > 0:
        # Transform for input format of the model
        X_keeping = numpy.transpose(X_keeping, (0, 2, 1, 3))
        # Output (predictions on analogies)
        y_keeping_pred = conv_model.predict(X_keeping, verbose=0)
        if voting == "majority":
            keep_counter = (y_keeping_pred >= voting_threshold).astype('int32')
            keep += numpy.count_nonzero(keep_counter == 1)
            prune += numpy.count_nonzero(keep_counter == 0)
        if voting == "weighted":
            keep = y_keeping_pred
        # Analogies stats
        for pred in y_keeping_pred:
            analogy_stats.confidence.append(pred[0])

    if len(X_pruning) > 0:
        # Transform for input format of the model
        X_pruning = numpy.transpose(X_pruning, (0, 2, 1, 3))
        # Output (predictions on analogies)
        y_pruning_pred = conv_model.predict(X_pruning, verbose=0)
        if voting == "majority":
            prune_counter = (y_pruning_pred < voting_threshold).astype('int32')
            prune += numpy.count_nonzero(prune_counter == 1)
            keep += numpy.count_nonzero(prune_counter == 0)
        if voting == "weighted":
            prune = y_pruning_pred
        # Analogies stats
        for pred in y_pruning_pred:
            analogy_stats.confidence.append(pred[0])

    # Voting
    if voting == "weighted":
        if "kk" in valid and "pp" not in valid:
            keep = numpy.mean(keep)
        elif "pp" in valid and "kk" not in valid:
            if len(X_pruning) > 0:
                keep = numpy.mean(1-prune)
        elif "pp" in valid and "kk" in valid:
            if len(X_pruning) > 0:
                keep = numpy.mean(numpy.concatenate((keep, 1 - prune)))
            else:
                keep = numpy.mean(keep)
        if keep >= voting_threshold:
            keeping_decisions.append(decision)
            # Analogies stats
            analogy_stats.append_stats_in_voting(target, 1, len(X_keeping)+len(X_pruning))
            return 1
        else:
            pruning_decisions.append(decision)
            # Analogies stats
            analogy_stats.append_stats_in_voting(target, 0, len(X_keeping)+len(X_pruning))
            return 0  
    elif voting == "majority":
        if keep >= voting_threshold:
            keeping_decisions.append(decision)
            # Analogies stats
            analogy_stats.append_stats_in_voting(target, 1, len(X_keeping)+len(X_pruning))
            return 1
        else:
            pruning_decisions.append(decision)
            # Analogies stats
            analogy_stats.append_stats_in_voting(target, 0, len(X_keeping)+len(X_pruning))
            return 0
        

def analogies_voting2(conv_model, 
                     voting,
                     voting_threshold,
                     X_keeping,
                     X_pruning,
                     valid,
                     analogy_stats,
                     target,
                     decision=[],
                     keeping_decisions=[],
                     pruning_decisions=[]):
    """ Voting of analogies to make a keeping decision or pruning decision """

    # Analogy predictions
    keep = 0
    prune = 0

    keep_counter = []
    prune_counter = []


    if len(X_keeping) > 0:
        # Transform for input format of the model
        X_keeping = numpy.transpose(X_keeping, (0, 2, 1, 3))
        # Output (predictions on analogies)
        #y_keeping_pred = conv_model.predict(X_keeping, verbose=0)
        # MC Dropout
        y_keeping_pred = numpy.stack([conv_model(X_keeping, training=True) for sample in range(100)])
        y_keeping_pred = y_keeping_pred.mean(axis=0)
        if voting == "majority":
            keep_counter = (y_keeping_pred >= voting_threshold).astype('int32')
            keep += numpy.count_nonzero(keep_counter == 1)
            prune += numpy.count_nonzero(keep_counter == 0)
        if voting == "weighted":
            keep = y_keeping_pred
        # Analogies stats
        for pred in y_keeping_pred:
            analogy_stats.confidence.append(pred[0])

    if len(X_pruning) > 0:
        # Transform for input format of the model
        X_pruning = numpy.transpose(X_pruning, (0, 2, 1, 3))
        # Output (predictions on analogies)
        #y_pruning_pred = conv_model.predict(X_pruning, verbose=0)
        y_pruning_pred = numpy.stack([conv_model(X_pruning, training=True) for sample in range(100)])
        y_pruning_pred = y_pruning_pred.mean(axis=0)
        if voting == "majority":
            prune_counter = (y_pruning_pred < voting_threshold).astype('int32')
            prune += numpy.count_nonzero(prune_counter == 1)
            keep += numpy.count_nonzero(prune_counter == 0)
        if voting == "weighted":
            prune = y_pruning_pred
        # Analogies stats
        for pred in y_pruning_pred:
            analogy_stats.confidence.append(pred[0])

    # Voting
    if voting == "weighted":
        if "kk" in valid and "pp" not in valid:
            keep = numpy.mean(keep)
        elif "pp" in valid and "kk" not in valid:
            if len(X_pruning) > 0:
                keep = numpy.mean(1-prune)
        elif "pp" in valid and "kk" in valid:
            if len(X_pruning) > 0:
                keep = numpy.mean(numpy.concatenate((keep, 1 - prune)))
            else:
                keep = numpy.mean(keep)
        if keep >= voting_threshold:
            keeping_decisions.append(decision)
            # Analogies stats
            analogy_stats.append_stats_in_voting(target, 1, len(X_keeping)+len(X_pruning))
            return 1
        else:
            pruning_decisions.append(decision)
            # Analogies stats
            analogy_stats.append_stats_in_voting(target, 0, len(X_keeping)+len(X_pruning))
            return 0  
    elif voting == "majority":
        if keep >= voting_threshold:
            keeping_decisions.append(decision)
            # Analogies stats
            analogy_stats.append_stats_in_voting(target, 1, len(X_keeping)+len(X_pruning))
            return 1
        else:
            pruning_decisions.append(decision)
            # Analogies stats
            analogy_stats.append_stats_in_voting(target, 0, len(X_keeping)+len(X_pruning))
            return 0


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
                           decision_train,
                           decision_to_test, 
                           vec_AB, 
                           vec_CD, 
                           analogies_stats,
                           known_decision,
                           properties):
    
    test_set.append(vec_AB + vec_CD)
    analogies_stats.append_stats_in_generation(decision_to_test[0], 
                                                               decision_to_test[3], 
                                                               decision_to_test[1],
                                                               decision_to_test[4],
                                                               decision_train[0],
                                                               decision_train[3],
                                                               decision_train[1],
                                                               decision_train[4],
                                                               abs(decision_train[2]-decision_to_test[2]),
                                                               known_decision, 1)
    if "symmetry" in properties:
        test_set.append(vec_CD + vec_AB)
        analogies_stats.append_stats_in_generation(decision_to_test[0], 
                                                               decision_to_test[3], 
                                                               decision_to_test[1],
                                                               decision_to_test[4],
                                                               decision_train[0],
                                                               decision_train[3],
                                                               decision_train[1],
                                                               decision_train[4],
                                                               abs(decision_train[2]-decision_to_test[2]),
                                                               known_decision, 1)
    if "inner-symmetry" in properties:
        vec_AB.reverse()
        vec_CD.reverse()
        test_set.append(vec_AB + vec_CD)
        analogies_stats.append_stats_in_generation(decision_to_test[0], 
                                                               decision_to_test[3], 
                                                               decision_to_test[1],
                                                               decision_to_test[4],
                                                               decision_train[0],
                                                               decision_train[3],
                                                               decision_train[1],
                                                               decision_train[4],
                                                               abs(decision_train[2]-decision_to_test[2]),
                                                               known_decision, 1)
        if "symmetry" in properties:
            test_set.append(vec_CD + vec_AB)
            analogies_stats.append_stats_in_generation(decision_to_test[0], 
                                                               decision_to_test[3], 
                                                               decision_to_test[1],
                                                               decision_to_test[4],
                                                               decision_train[0],
                                                               decision_train[3],
                                                               decision_train[1],
                                                               decision_train[4],
                                                               abs(decision_train[2]-decision_to_test[2]),
                                                               known_decision, 1)
        vec_AB.reverse()
        vec_CD.reverse()


def add_evaluation_analogy(test_set, 
                           decision_train,
                           decision_to_test, 
                           vec_A,
                           vec_B, 
                           vec_C,
                           vec_D, 
                           analogies_stats,
                           known_decision,
                           properties):
    
    test_set.append([vec_A, vec_B, vec_C, vec_D])
    analogies_stats.append_stats_in_generation(decision_to_test[0], 
                                                               decision_to_test[3], 
                                                               decision_to_test[1],
                                                               decision_to_test[4],
                                                               decision_train[0],
                                                               decision_train[3],
                                                               decision_train[1],
                                                               decision_train[4],
                                                               abs(decision_train[2]-decision_to_test[2]),
                                                               known_decision, 1)
    if "symmetry" in properties:
        test_set.append([vec_C, vec_D, vec_A, vec_B])
        analogies_stats.append_stats_in_generation(decision_to_test[0], 
                                                               decision_to_test[3], 
                                                               decision_to_test[1],
                                                               decision_to_test[4],
                                                               decision_train[0],
                                                               decision_train[3],
                                                               decision_train[1],
                                                               decision_train[4],
                                                               abs(decision_train[2]-decision_to_test[2]),
                                                               known_decision, 1)
    if "inner-symmetry" in properties:
        test_set.append([vec_B, vec_A, vec_D, vec_C])
        analogies_stats.append_stats_in_generation(decision_to_test[0], 
                                                               decision_to_test[3], 
                                                               decision_to_test[1],
                                                               decision_to_test[4],
                                                               decision_train[0],
                                                               decision_train[3],
                                                               decision_train[1],
                                                               decision_train[4],
                                                               abs(decision_train[2]-decision_to_test[2]),
                                                               known_decision, 1)
        if "symmetry" in properties:
            test_set.append([vec_D, vec_C, vec_B, vec_A])
            analogies_stats.append_stats_in_generation(decision_to_test[0], 
                                                               decision_to_test[3], 
                                                               decision_to_test[1],
                                                               decision_to_test[4],
                                                               decision_train[0],
                                                               decision_train[3],
                                                               decision_train[1],
                                                               decision_train[4],
                                                               abs(decision_train[2]-decision_to_test[2]),
                                                               known_decision, 1)
