"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import random
import argparse
import time
import csv
import json

import pandas
import numpy
import lmdb
import tensorflow as tf

import utils
from TqdmLoggingHandler import *


def analogy_model(nb_filters1, nb_filters2, dropout):
    """ Build the analogy model """
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=nb_filters1, 
                                   kernel_size=(1,2), 
                                   strides=(1,2), 
                                   activation="relu", 
                                   input_shape=(200, 4, 1), 
                                   kernel_initializer="he_normal"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Conv2D(filters=nb_filters2,
                                   kernel_size=(2,2), 
                                   strides=(2,2), 
                                   activation="relu", 
                                   input_shape=(200, 2, nb_filters1), 
                                   kernel_initializer="he_normal"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1, activation="sigmoid")
        ])
    return model


def transpose(embedding):
    return numpy.transpose(numpy.array([embedding]), (1, 0))


def get_analogies(train_features, 
                  train_labels, 
                  keeping_train, 
                  pruning_train, 
                  nb_training_analogies_per_decision, 
                  embedding_hashmap,
                  properties,
                  valid_analogies_pattern,
                  invalid_analogies_pattern,
                  knn):
    
    keeping_decisions = keeping_train.values.tolist()
    pruning_decisions = pruning_train.values.tolist()
    keeping_to_shuffle = keeping_decisions.copy()
    pruning_to_shuffle = pruning_decisions.copy()

    # Keeping decisions :: Keeping decisions
    for i in range(len(keeping_decisions)):
        vec_kept_A = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping_decisions[i][0] + '>', embedding_hashmap)
        vec_kept_B = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping_decisions[i][1] + '>', embedding_hashmap)
        if vec_kept_A is not None and vec_kept_B is not None:
            vec_kept_A = transpose(vec_kept_A)
            vec_kept_B = transpose(vec_kept_B)
            if "kk" in valid_analogies_pattern + invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0

                keeping = []
                if "train" not in knn:
                    random.shuffle(keeping_to_shuffle)
                    keeping = keeping_to_shuffle
                else:
                    # K nearest neighbors
                    keeping = utils.get_hashmap_distances(keeping_decisions[i][0],
                                                          keeping_train,
                                                          embedding_hashmap)

                while nb_used_analogies < nb_training_analogies_per_decision:
                    if training_analogy_index == len(keeping):
                        break
                    vec_kept_C = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping[training_analogy_index][0] + '>', embedding_hashmap)
                    vec_kept_D = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping[training_analogy_index][1] + '>', embedding_hashmap)
                    if vec_kept_C is not None and vec_kept_D is not None and keeping[training_analogy_index][0] != keeping_decisions[i][0]:
                        vec_kept_C = transpose(vec_kept_C)
                        vec_kept_D = transpose(vec_kept_D)
                        utils.add_analogy("train", vec_kept_C, vec_kept_D, vec_kept_A, vec_kept_B, properties, valid_analogies_pattern, "kk", train_features, train_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1
            
            if "pk" in valid_analogies_pattern + invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0

                pruning = []
                if "train" not in knn:
                    random.shuffle(pruning_to_shuffle)
                    pruning = pruning_to_shuffle
                else:
                    # K nearest neighbors
                    pruning = utils.get_hashmap_distances(keeping_decisions[i][0],
                                                          pruning_train,
                                                          embedding_hashmap)
                # Invalid analogies
                while nb_used_analogies < nb_training_analogies_per_decision:
                    if training_analogy_index == len(pruning):
                        break
                    vec_pruned_C = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning[training_analogy_index][0] + '>', embedding_hashmap)
                    vec_pruned_D = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning[training_analogy_index][1] + '>', embedding_hashmap)
                    if vec_pruned_C is not None and vec_pruned_D is not None and pruning[0] != keeping_decisions[i][0]:
                        vec_pruned_C = transpose(vec_pruned_C)
                        vec_pruned_D = transpose(vec_pruned_D)
                        # Pruning decisions :: Keeping decisions
                        utils.add_analogy("train", vec_pruned_C, vec_pruned_D, vec_kept_A, vec_kept_B, properties, valid_analogies_pattern, "pk", train_features, train_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1
            
    # Pruning decisions :: Pruning decisions
    for i in range(len(pruning_decisions)):
        vec_pruned_A = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning_decisions[i][0] + '>', embedding_hashmap)
        vec_pruned_B = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning_decisions[i][1] + '>', embedding_hashmap)
        if vec_pruned_A is not None and vec_pruned_B is not None:
            vec_pruned_A = transpose(vec_pruned_A)
            vec_pruned_B = transpose(vec_pruned_B)
            if "pp" in valid_analogies_pattern + invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0

                pruning = []
                if "train" not in knn:
                    random.shuffle(pruning_to_shuffle)
                    pruning = pruning_to_shuffle
                else:
                    # K nearest neighbors
                    pruning = utils.get_hashmap_distances(pruning_decisions[i][0],
                                                          pruning_train,
                                                          embedding_hashmap)

                while nb_used_analogies < nb_training_analogies_per_decision:
                    if training_analogy_index == len(pruning):
                        break
                    vec_pruned_C = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning[training_analogy_index][0] + '>', embedding_hashmap)
                    vec_pruned_D = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning[training_analogy_index][1] + '>', embedding_hashmap)
                    if vec_pruned_C is not None and vec_pruned_D is not None and pruning[0] != pruning_decisions[i][0]:
                        vec_pruned_C = transpose(vec_pruned_C)
                        vec_pruned_D = transpose(vec_pruned_D)
                        utils.add_analogy("train", vec_pruned_C, vec_pruned_D, vec_pruned_A, vec_pruned_B, properties, valid_analogies_pattern, "pp", train_features, train_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1
            
            if "kp" in valid_analogies_pattern + invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0

                keeping = []
                if "train" not in knn:
                    random.shuffle(keeping_to_shuffle)
                    keeping = keeping_to_shuffle
                else:
                    # K nearest neighbors
                    keeping = utils.get_hashmap_distances(pruning_decisions[i][0],
                                                          keeping_train,
                                                          embedding_hashmap)

                # Invalid analogies
                while nb_used_analogies < nb_training_analogies_per_decision:
                    if training_analogy_index == len(keeping):
                        break
                    vec_kept_C = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping[training_analogy_index][0] + '>', embedding_hashmap)
                    vec_kept_D = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping[training_analogy_index][1] + '>', embedding_hashmap)
                    if vec_kept_C is not None and vec_kept_D is not None and keeping[0] != pruning_decisions[i][0]:
                        vec_kept_C = transpose(vec_kept_C)
                        vec_kept_D = transpose(vec_kept_D)
                        # Keeping decisions :: Pruning decisions
                        utils.add_analogy("train", vec_kept_C, vec_kept_D, vec_pruned_A, vec_pruned_B, properties, valid_analogies_pattern, "kp", train_features, train_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1


def get_testing_analogies(test_features, 
                          test_labels, 
                          keeping_decisions_test, 
                          pruning_decisions_test,
                          keeping_train,
                          pruning_train, 
                          nb_test_analogies, 
                          embedding_hashmap,
                          properties,
                          valid_analogies_pattern,
                          invalid_analogies_pattern,
                          knn):
    
    keeping_decisions = keeping_train.values.tolist()
    pruning_decisions = pruning_train.values.tolist()
    keeping_to_shuffle = keeping_decisions.copy()
    pruning_to_shuffle = pruning_decisions.copy()
    
    # Keeping decisions :: Keeping decisions
    for i in range(len(keeping_decisions_test)):
        vec_kept_A = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping_decisions_test[i][0] + '>', embedding_hashmap)
        vec_kept_B = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping_decisions_test[i][1] + '>', embedding_hashmap)
        if vec_kept_A is not None and vec_kept_B is not None:
            vec_kept_A = transpose(vec_kept_A)
            vec_kept_B = transpose(vec_kept_B)
            if "kk" in valid_analogies_pattern + invalid_analogies_pattern and "kp" in invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0

                keeping = []
                if "test" not in knn:
                    random.shuffle(keeping_to_shuffle)
                    keeping = keeping_to_shuffle
                else:
                    # K nearest neighbors
                    keeping = utils.get_hashmap_distances(keeping_decisions_test[i][0],
                                                          keeping_train,
                                                          embedding_hashmap)

                while nb_used_analogies < nb_test_analogies:
                    if training_analogy_index == len(keeping):
                        break
                    vec_kept_C = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping[training_analogy_index][0] + '>', embedding_hashmap)
                    vec_kept_D = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping[training_analogy_index][1] + '>', embedding_hashmap)
                    if vec_kept_C is not None and vec_kept_D is not None:
                        vec_kept_C = transpose(vec_kept_C)
                        vec_kept_D = transpose(vec_kept_D)
                        utils.add_analogy("test", vec_kept_C, vec_kept_D, vec_kept_A, vec_kept_B, properties, valid_analogies_pattern, "kk", test_features, test_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1
            
            if "pk" in valid_analogies_pattern + invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0

                pruning = []
                if "test" not in knn:
                    random.shuffle(pruning_to_shuffle)
                    pruning = pruning_to_shuffle
                else:
                    # K nearest neighbors
                    pruning = utils.get_hashmap_distances(keeping_decisions_test[i][0],
                                                          pruning_train,
                                                          embedding_hashmap)

                while nb_used_analogies < nb_test_analogies:
                    if training_analogy_index == len(pruning):
                        break
                    vec_pruned_C = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning[training_analogy_index][0] + '>', embedding_hashmap)
                    vec_pruned_D = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning[training_analogy_index][1] + '>', embedding_hashmap)
                    if vec_pruned_C is not None and vec_pruned_D is not None:
                        vec_pruned_C = transpose(vec_pruned_C)
                        vec_pruned_D = transpose(vec_pruned_D)
                        # Pruning decisions :: Keeping decisions
                        utils.add_analogy("test", vec_pruned_C, vec_pruned_D, vec_kept_A, vec_kept_B, properties, valid_analogies_pattern, "pk", test_features, test_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1
            
    # Pruning decisions :: Pruning decisions
    for i in range(len(pruning_decisions_test)):
        vec_pruned_A = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning_decisions_test[i][0] + '>', embedding_hashmap)
        vec_pruned_B = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning_decisions_test[i][1] + '>', embedding_hashmap)
        if vec_pruned_A is not None and vec_pruned_B is not None:
            vec_pruned_A = transpose(vec_pruned_A)
            vec_pruned_B = transpose(vec_pruned_B)
            if "pp" in valid_analogies_pattern + invalid_analogies_pattern and "pk" in invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0

                pruning = []
                if "test" not in knn:
                    random.shuffle(pruning_to_shuffle)
                    pruning = pruning_to_shuffle
                else:
                    # K nearest neighbors
                    pruning = utils.get_hashmap_distances(pruning_decisions_test[i][0],
                                                          pruning_train,
                                                          embedding_hashmap)

                while nb_used_analogies < nb_test_analogies:
                    if training_analogy_index == len(pruning):
                        break
                    vec_pruned_C = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning[training_analogy_index][0] + '>', embedding_hashmap)
                    vec_pruned_D = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning[training_analogy_index][1] + '>', embedding_hashmap)
                    if vec_pruned_C is not None and vec_pruned_D is not None:
                        vec_pruned_C = transpose(vec_pruned_C)
                        vec_pruned_D = transpose(vec_pruned_D)
                        utils.add_analogy("test", vec_pruned_C, vec_pruned_D, vec_pruned_A, vec_pruned_B, properties, valid_analogies_pattern, "pp", test_features, test_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1
    
            if "kp" in valid_analogies_pattern + invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0

                keeping = []
                if "test" not in knn:
                    random.shuffle(keeping_to_shuffle)
                    keeping = keeping_to_shuffle
                else:
                    # K nearest neighbors
                    keeping = utils.get_hashmap_distances(pruning_decisions_test[i][0],
                                                          keeping_train,
                                                          embedding_hashmap)

                # Invalid analogies
                while nb_used_analogies < nb_test_analogies:
                    if training_analogy_index == len(keeping):
                        break
                    vec_kept_C = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping[training_analogy_index][0] + '>', embedding_hashmap)
                    vec_kept_D = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping[training_analogy_index][1] + '>', embedding_hashmap)
                    if vec_kept_C is not None and vec_kept_D is not None:
                        vec_kept_C = transpose(vec_kept_C)
                        vec_kept_D = transpose(vec_kept_D)
                        # Keeping decisions :: Pruning decisions
                        utils.add_analogy("test", vec_kept_C, vec_kept_D, vec_pruned_A, vec_pruned_B, properties, valid_analogies_pattern, "kp", test_features, test_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1


def get_evaluation_analogies(qid,
                             keeping_train,
                             pruning_train, 
                             nb_keeping_in_test,
                             nb_pruning_in_test,
                             embedding_hashmap,
                             properties,
                             valid_analogies_pattern,
                             invalid_analogies_pattern,
                             knn,
                             classes):
    
    keeping_test = []
    pruning_test = []
    classes_test = []

    keeping_decisions = keeping_train.values.tolist()
    pruning_decisions = pruning_train.values.tolist()
    keeping_to_shuffle = keeping_decisions.copy()
    pruning_to_shuffle = pruning_decisions.copy()

    for cl in classes:
        decision_to_test = [qid, cl]
        # Decision to test :: Keeping decisions
        vec_kept_A = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + decision_to_test[0] + '>', embedding_hashmap)
        vec_kept_B = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + decision_to_test[1] + '>', embedding_hashmap)
        if vec_kept_A is not None and vec_kept_B is not None:
            vec_kept_A = transpose(vec_kept_A)
            vec_kept_B = transpose(vec_kept_B)
            if "kk" in valid_analogies_pattern + invalid_analogies_pattern and "kp" in invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0

                keeping = []
                if "test" not in knn:
                    random.shuffle(keeping_to_shuffle)
                    keeping = keeping_to_shuffle
                else:
                    # K nearest neighbors
                    keeping = utils.get_hashmap_distances(decision_to_test[0],
                                                        keeping_train,
                                                        embedding_hashmap)

                while nb_used_analogies < nb_keeping_in_test:
                    if training_analogy_index == len(keeping):
                        break
                    vec_kept_C = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping[training_analogy_index][0] + '>', embedding_hashmap)
                    vec_kept_D = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + keeping[training_analogy_index][1] + '>', embedding_hashmap)
                    if vec_kept_C is not None and vec_kept_D is not None:
                        vec_kept_C = transpose(vec_kept_C)
                        vec_kept_D = transpose(vec_kept_D)
                        utils.add_evaluation_analogy(keeping_test,
                                            vec_kept_C,
                                            vec_kept_D,
                                            vec_kept_A,
                                            vec_kept_B,
                                            properties)
                        nb_used_analogies += 1
                    training_analogy_index += 1

        # Decision to test :: Pruning decisions
        vec_pruned_A = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + decision_to_test[0] + '>', embedding_hashmap)
        vec_pruned_B = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + decision_to_test[1] + '>', embedding_hashmap)
        if vec_pruned_A is not None and vec_pruned_B is not None:
            vec_pruned_A = transpose(vec_pruned_A)
            vec_pruned_B = transpose(vec_pruned_B)
            if "pp" in valid_analogies_pattern + invalid_analogies_pattern and "pk" in invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0

                pruning = []
                if "test" not in knn:
                    random.shuffle(pruning_to_shuffle)
                    pruning = pruning_to_shuffle
                else:
                    # K nearest neighbors
                    pruning = utils.get_hashmap_distances(decision_to_test[0],
                                                        pruning_train,
                                                        embedding_hashmap)

                while nb_used_analogies < nb_pruning_in_test:
                    if training_analogy_index == len(pruning):
                        break
                    vec_pruned_C = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning[training_analogy_index][0] + '>', embedding_hashmap)
                    vec_pruned_D = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + pruning[training_analogy_index][1] + '>', embedding_hashmap)
                    if vec_pruned_C is not None and vec_pruned_D is not None:
                        vec_pruned_C = transpose(vec_pruned_C)
                        vec_pruned_D = transpose(vec_pruned_D)
                        utils.add_evaluation_analogy(pruning_test,
                                            vec_pruned_C,
                                            vec_pruned_D,
                                            vec_pruned_A,
                                            vec_pruned_B,
                                            properties)
                        nb_used_analogies += 1
                    training_analogy_index += 1
        
    return keeping_test, pruning_test, classes_test


def main():

    parser = argparse.ArgumentParser(prog="analogy_pruning", description="Analogy-based classifier")
    parser.add_argument("--voting", help="Voting method for expansion (evaluation of analogy model)", required=False, default="weighted")
    parser.add_argument("--voting-threshold", dest="voting_threshold", help="Float value representing the threshold for deciding whether prediction analogies are 1 or 0",
                         required=False, default=0.5, type=float)
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--knn", required=False, default=["train test"], nargs="+")
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--nb-training-analogies-per-decision", dest="nb_training_analogies_per_decision", 
                        required=False, default=20, type=int, help="Number of training analogies per decision")
    parser.add_argument("--nb-test-analogies", dest="nb_test_analogies", type=int, required=False, help="Number of test analogies", default=20)
    parser.add_argument("--nb-keeping-in-test", dest="nb_keeping_in_test", type=int, required=False, help="Number of keeping test analogies", default=20)
    parser.add_argument("--nb-pruning-in-test", dest="nb_pruning_in_test", type=int, required=False, help="Number of pruning test analogies", default=20)
    parser.add_argument("--analogical-properties", dest="analogical_properties", required=False, default=["."], nargs='+')
    parser.add_argument("--valid-analogies-pattern", dest="valid_analogies_pattern", nargs='+')
    parser.add_argument("--invalid-analogies-pattern", dest="invalid_analogies_pattern", nargs='+')
    parser.add_argument("--node-degree", dest="node_degree", required=False, default=100_000_000, type=int)
    parser.add_argument("--analogy-batch", dest="analogy_batch", required=False, default=1_000, type=int)
    parser.add_argument("--seed-qids", dest="seed_qids", help="CSV file containing QIDs of interest (one QID per line)", required=True)
    parser.add_argument("--output-statistics", dest="output_statistics", help="Output file to save statistics", required=True)
    parser.add_argument("--relevant-entities", dest="relevant_entities", help="Relevant entities selected from seed QIDs of interest", required=True)
    parser.add_argument("--nb-filters1", dest="nb_filters1", required=False, default=16, type=int)
    parser.add_argument("--nb-filters2", dest="nb_filters2", required=False, default=8, type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", help="Learning rate", required=False, default=1e-4, type=float)
    parser.add_argument("--dropout", required=False, default=0.1, type=float)
    parser.add_argument("--epochs", help="Number of epochs to train models", required=False, default=50, type=int)
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

    # Load QIDs of interest
    qids_of_interest = set()
    csv_file = open(args.seed_qids, 'r')
    for line in csv_file:
        qid = line.split('\n')[0]
        qids_of_interest.add(qid)
    csv_file.close()
    qids_of_interest = qids_of_interest

    stats_seen_classes = set()

    train_set = set(decisions["from"])
    train_set_list = list(train_set)
    random.shuffle(train_set_list)
    val_set = set(train_set_list[:int(len(train_set_list) / 5)])
    train_set -= val_set

    val_features = []
    val_labels = []

    train_features = []
    train_labels = []

    kept_classes = dict()

    logger.info("Building training data")
    keeping_decisions_train = decisions[(decisions["from"].isin(train_set)) & (decisions["target"] == 1)]
    pruning_decisions_train = decisions[(decisions["from"].isin(train_set)) & (decisions["target"] == 0)]

    # Generate analogies from keeping and pruning decisions
    get_analogies(train_features,
                train_labels, 
                keeping_decisions_train[["from", "QID", "depth", "starting label", "label"]], 
                pruning_decisions_train[["from", "QID", "depth", "starting label", "label"]], 
                args.nb_training_analogies_per_decision,
                embedding_hashmap,
                args.analogical_properties,
                args.valid_analogies_pattern,
                args.invalid_analogies_pattern,
                args.knn)
    

    logger.info("Building validation analogies")
    keeping_decisions_val = decisions[(decisions["from"].isin(val_set)) & (decisions["target"] == 1)]
    pruning_decisions_val = decisions[(decisions["from"].isin(val_set)) & (decisions["target"] == 0)]

    # Generate analogies from keeping and pruning decisions
    get_testing_analogies(val_features,
                          val_labels,
                          keeping_decisions_val[["from", "QID", "depth"]].values.tolist(), 
                          pruning_decisions_val[["from", "QID", "depth"]].values.tolist(),
                          keeping_decisions_train[["from", "QID", "depth", "starting label", "label"]],
                          pruning_decisions_train[["from", "QID", "depth", "starting label", "label"]],
                          args.nb_test_analogies, 
                          embedding_hashmap,
                          args.analogical_properties,
                          args.valid_analogies_pattern,
                          args.invalid_analogies_pattern,
                          args.knn)
    
    val_features = numpy.transpose(val_features, (0, 2, 1, 3))
    val_labels = numpy.array(val_labels)

    # Tranform to the input format of analogy-based convolutional model
    train_features = numpy.transpose(train_features, (0, 2, 1, 3))
    train_labels = numpy.array(train_labels)

    # Training
    analogy_classifier = analogy_model(args.nb_filters1, args.nb_filters2, args.dropout)
    analogy_classifier.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                            metrics=[tf.keras.metrics.Precision(), 
                                    tf.keras.metrics.Recall(), 
                                    tf.keras.metrics.BinaryAccuracy()])

    weight_for_0 = 1 / numpy.count_nonzero(train_labels == 0) * (len(train_labels) / 2.0)
    weight_for_1 = 1 / numpy.count_nonzero(train_labels == 1) * (len(train_labels) / 2.0)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")

    analogy_classifier.fit(train_features,
                           train_labels,
                           validation_data=(val_features, val_labels),
                           callbacks=[early_stopping_cb],
                           class_weight={0: weight_for_0, 1: weight_for_1},
                           shuffle=True,
                           epochs=args.epochs)

    logger.info("Expansion --> evaluation of decisions")
    # Start the expansion
    for qid in tqdm.tqdm(qids_of_interest):
        seen_classes = {qid}
        depth = 0
        qid_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + qid + ">", embedding_hashmap)
        kept_classes[qid] = []

        if qid_emb is not None:

            q_adjacency = utils.get_hashmap_content(qid, wikidata_hashmap)

            if q_adjacency is not None and ("P31" in q_adjacency["claims"] or "P279" in q_adjacency["claims"] or
                                "(-)P279" in q_adjacency["claims"]):
                classes = set()

                if "P31" in q_adjacency["claims"]:
                    classes |= utils.get_relation_objects(q_adjacency["claims"]["P31"])
                if "P279" in q_adjacency["claims"]:
                    classes |= utils.get_relation_objects(q_adjacency["claims"]["P279"])
                if "(-)P279" in q_adjacency["claims"]:
                    classes |= utils.get_relation_objects(q_adjacency["claims"]["(-)P279"])

                while classes:
                    seen_classes |= classes
                    new_classes = set()
                    depth += 1

                    for cl in classes:
                        cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)
                        if cl_emb is not None:

                            # Batch processing
                            classes_list = list(classes)
                            classes_batch = lambda classes_list, batch: [classes_list[i:i+batch] for i in range(0, len(classes_list), batch)]
                            classes_list = classes_batch(classes_list, args.analogy_batch)

                            for cl_list in classes_list:

                                # Generate analogies
                                keeping_test_analogies, pruning_test_analogies, classes_test = get_evaluation_analogies(qid,
                                                                                                        keeping_decisions_train[["from", "QID", "depth", "starting label", "label"]],
                                                                                                        pruning_decisions_train[["from", "QID", "depth", "starting label", "label"]],
                                                                                                        args.nb_keeping_in_test,
                                                                                                        args.nb_pruning_in_test,
                                                                                                        embedding_hashmap,
                                                                                                        args.analogical_properties,
                                                                                                        args.valid_analogies_pattern,
                                                                                                        args.invalid_analogies_pattern,
                                                                                                        args.knn,
                                                                                                        cl_list)
                                
                                predictions = dict()
                                
                                if len(keeping_test_analogies) > 0:
                                    # Transform for input format of the model
                                    keeping_test_analogies = numpy.transpose(keeping_test_analogies, (0, 2, 1, 3))
                                    # Output (predictions on analogies)
                                    y_keeping_pred = 0
                                    if args.dropout > 0:
                                        probas = numpy.stack([analogy_classifier(keeping_test_analogies, training=True) for sample in range(10)])
                                        y_keeping_pred = probas.mean(axis=0)
                                    else:
                                        y_keeping_pred = analogy_classifier.predict(keeping_test_analogies, verbose=0)
                                    predictions["keeping"] = y_keeping_pred
                                        
                                if len(pruning_test_analogies) > 0:
                                    # Transform for input format of the model
                                    pruning_test_analogies = numpy.transpose(pruning_test_analogies, (0, 2, 1, 3))
                                    # Output (predictions on analogies)
                                    y_pruning_pred = 0
                                    if args.dropout > 0:
                                        probas = numpy.stack([analogy_classifier(pruning_test_analogies, training=True) for sample in range(10)])
                                        y_pruning_pred = probas.mean(axis=0)
                                    else:
                                        y_pruning_pred = analogy_classifier.predict(pruning_test_analogies, verbose=0)
                                    predictions["pruning"] = y_pruning_pred

                                for i, cl in enumerate(classes_test):

                                    decision = 0
                                    # Analogy predictions
                                    keep = 0
                                    prune = 0

                                    keep_counter = []
                                    prune_counter = []

                                    if "keeping" in predictions:
                                        y_keeping_pred = predictions["keeping"][i*args.nb_keeping_in_test:(i+1)*args.nb_keeping_in_test]
                                        if args.voting == "majority":
                                            keep_counter = (y_keeping_pred >= args.voting_threshold).astype('int32')
                                            keep += numpy.count_nonzero(keep_counter == 1)
                                            prune += numpy.count_nonzero(keep_counter == 0)
                                        if args.voting == "weighted":
                                            keep = y_keeping_pred

                                    if "pruning" in predictions:
                                        y_pruning_pred = predictions["pruning"][i*args.nb_pruning_in_test:(i+1)*args.nb_pruning_in_test]
                                        if args.voting == "majority":
                                            prune_counter = (y_pruning_pred < args.voting_threshold).astype('int32')
                                            prune += numpy.count_nonzero(prune_counter == 1)
                                            keep += numpy.count_nonzero(prune_counter == 0)
                                        if args.voting == "weighted":
                                            prune = y_pruning_pred
                                    # Voting
                                    if args.voting == "weighted":
                                        if "kk" in args.valid_analogies_pattern and "pp" not in args.valid_analogies_pattern:
                                            keep = numpy.mean(keep)
                                        elif "pp" in args.valid_analogies_pattern and "kk" not in args.valid_analogies_pattern:
                                            if "pruning" in predictions:
                                                keep = numpy.mean(1-prune)
                                        elif "pp" in args.valid_analogies_pattern and "kk" in args.valid_analogies_pattern:
                                            if "pruning" in predictions:
                                                keep = numpy.mean(numpy.concatenate((keep, 1 - prune)))
                                            else:
                                                keep = numpy.mean(keep)
                                        if keep >= args.voting_threshold:
                                            decision = 1
                                        else:
                                            decision = 0  
                                    elif args.voting == "majority":
                                        if keep >= args.voting_threshold:
                                            decision = 1
                                        else:
                                            decision = 0
                                    
                                    cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)
                                    if decision == 1 and cl_adjacency is not None:
                                        kept_classes[qid].append(cl)
                                        if utils.get_node_degree(cl_adjacency) < args.node_degree:
                                            new_classes |= utils.get_unseen_subclasses(cl_adjacency, seen_classes)
                                            stats_seen_classes.add(cl)

                    classes = new_classes

    with open(args.relevant_entities, "w") as outfile:
        json.dump(kept_classes, outfile)

    # Statistics output
    print(f"# nodes in the output KG = {len(stats_seen_classes)}")
    with open(args.output_statistics, "w") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["# nodes in the output KG", len(stats_seen_classes)])

    logger.info(f"Execution time = {utils.convert(time.time() - start)}")

    wikidata_hashmap.close()
    embedding_hashmap.close()

if __name__ == '__main__':
    main()