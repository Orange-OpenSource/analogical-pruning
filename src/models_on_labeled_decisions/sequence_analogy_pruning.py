"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import random
import pickle
import argparse
import time

import lmdb
import numpy
import pandas
import tensorflow as tf

import AnalogyStatistics as AnalogyStatistics
import utils
from TqdmLoggingHandler import *


def sequenced_analogy_model(seq_len, nb_filters1, nb_filters2, dropout):
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=nb_filters1, 
                                   kernel_size=(1, seq_len), 
                                   strides=(1, seq_len), 
                                   activation="relu", 
                                   input_shape=(200, seq_len*2, 1), 
                                   kernel_initializer="he_normal"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Conv2D(filters=nb_filters2, 
                                   kernel_size=(2, 2), 
                                   strides=(2, 2), 
                                   activation="relu", 
                                   input_shape=(200, 2, nb_filters1), 
                                   kernel_initializer="he_normal"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])
    return model


def transpose(embedding):
    return numpy.transpose(numpy.array([embedding]), (1, 0))


def get_analogies(train_features, 
                  train_labels, 
                  keeping_train, 
                  pruning_train, 
                  nb_training_analogies_per_decision,
                  sequenced_decisions, 
                  sequence_length, 
                  padding,
                  properties,
                  valid_analogies_pattern,
                  invalid_analogies_pattern,
                  distances_hashmap,
                  knn):

    keeping_decisions = keeping_train.values.tolist()
    pruning_decisions = pruning_train.values.tolist()
    keeping_to_shuffle = keeping_decisions.copy()
    pruning_to_shuffle = pruning_decisions.copy()

    # Keeping decisions :: Keeping decisions
    for i in range(len(keeping_decisions)):
        starting_qid_A = keeping_decisions[i][0]
        reached_class_B = keeping_decisions[i][1]
        if starting_qid_A in sequenced_decisions and reached_class_B in sequenced_decisions[starting_qid_A]:
            vec_kept_AB = utils.pad_sequence(sequenced_decisions[starting_qid_A][reached_class_B], sequence_length, padding)
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
                                                          distances_hashmap,
                                                          keeping_train)

                while nb_used_analogies < nb_training_analogies_per_decision:
                    if training_analogy_index == len(keeping):
                        break
                    starting_qid_C = keeping[training_analogy_index][0]
                    reached_class_D = keeping[training_analogy_index][1]
                    if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C] and \
                        keeping[training_analogy_index][0] != keeping_decisions[i][0]:
                        vec_kept_CD = utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length, padding)
                        utils.add_sequence_analogy("train", vec_kept_CD, vec_kept_AB, properties, valid_analogies_pattern, "kk", train_features, train_labels)
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
                                                          distances_hashmap,
                                                          pruning_train)

                # Invalid analogies
                while nb_used_analogies < nb_training_analogies_per_decision:
                    if training_analogy_index == len(pruning):
                        break
                    starting_qid_C = pruning[training_analogy_index][0]
                    reached_class_D = pruning[training_analogy_index][1]
                    if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C] and \
                        pruning[training_analogy_index][0] != keeping_decisions[i][0]:
                        vec_pruned_CD = utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length, padding)
                        # Pruning decisions :: Keeping decisions
                        utils.add_sequence_analogy("train", vec_pruned_CD, vec_kept_AB, properties, valid_analogies_pattern, "pk", train_features, train_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1
            
    # Pruning decisions :: Pruning decisions
    for i in range(len(pruning_decisions)):
        starting_qid_A = pruning_decisions[i][0]
        reached_class_B = pruning_decisions[i][1]
        if starting_qid_A in sequenced_decisions and reached_class_B in sequenced_decisions[starting_qid_A]:
            vec_pruned_AB = utils.pad_sequence(sequenced_decisions[starting_qid_A][reached_class_B], sequence_length, padding)
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
                                                          distances_hashmap,
                                                          pruning_train)

                while nb_used_analogies < nb_training_analogies_per_decision: 
                    if training_analogy_index == len(pruning):
                        break
                    starting_qid_C = pruning[training_analogy_index][0]
                    reached_class_D = pruning[training_analogy_index][1]
                    if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C] and \
                        pruning[training_analogy_index][0] != pruning_decisions[i][0]:
                        vec_pruned_CD = utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length, padding)
                        utils.add_sequence_analogy("train", vec_pruned_CD, vec_pruned_AB, properties, valid_analogies_pattern, "pp", train_features, train_labels)
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
                                                          distances_hashmap,
                                                          keeping_train)

                # Invalid analogies
                while nb_used_analogies < nb_training_analogies_per_decision:
                    if training_analogy_index == len(keeping):
                        break
                    starting_qid_C = keeping[training_analogy_index][0]
                    reached_class_D = keeping[training_analogy_index][1]
                    if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C] and \
                        keeping[training_analogy_index][0] != pruning_decisions[i][0]:
                        vec_kept_CD = utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length, padding)
                        # Keeping decisions :: Pruning decisions
                        utils.add_sequence_analogy("train", vec_kept_CD, vec_pruned_AB, properties, valid_analogies_pattern, "kp", train_features, train_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1
            
                
def get_testing_analogies(test_features, 
                          test_labels, 
                          keeping_decisions_test, 
                          pruning_decisions_test,
                          keeping_train,
                          pruning_train, 
                          nb_test_analogies, 
                          sequenced_decisions,
                          sequence_length,
                          padding,
                          properties,
                          valid_analogies_pattern,
                          invalid_analogies_pattern,
                          distances_hashmap,
                          knn):
    
    keeping_decisions = keeping_train.values.tolist()
    pruning_decisions = pruning_train.values.tolist()
    keeping_to_shuffle = keeping_decisions.copy()
    pruning_to_shuffle = pruning_decisions.copy()
    
    # Keeping decisions :: Keeping decisions
    for i in range(len(keeping_decisions_test)):
        starting_qid_A = keeping_decisions_test[i][0]
        reached_class_B = keeping_decisions_test[i][1]
        if starting_qid_A in sequenced_decisions and reached_class_B in sequenced_decisions[starting_qid_A]:
            vec_kept_AB = utils.pad_sequence(sequenced_decisions[starting_qid_A][reached_class_B], sequence_length, padding)
            
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
                                                          distances_hashmap,
                                                          keeping_train)

                while nb_used_analogies < nb_test_analogies:
                    if training_analogy_index == len(keeping):
                        break
                    starting_qid_C = keeping[training_analogy_index][0]
                    reached_class_D = keeping[training_analogy_index][1]
                    if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C]:
                        vec_kept_CD = utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length, padding)
                        utils.add_sequence_analogy("test", vec_kept_CD, vec_kept_AB, properties, valid_analogies_pattern, "kk", test_features, test_labels)
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
                                                          distances_hashmap,
                                                          pruning_train)

                # Invalid analogies
                while nb_used_analogies < nb_test_analogies:
                    if training_analogy_index == len(pruning):
                        break
                    starting_qid_C = pruning[training_analogy_index][0]
                    reached_class_D = pruning[training_analogy_index][1]
                    if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C]:
                        vec_pruned_CD = utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length, padding)
                        # Pruning decisions :: Keeping decisions
                        utils.add_sequence_analogy("test", vec_pruned_CD, vec_kept_AB, properties, valid_analogies_pattern, "pk", test_features, test_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1
    
    # Pruning decisions :: Pruning decisions
    for i in range(len(pruning_decisions_test)):
        starting_qid_A = pruning_decisions_test[i][0]
        reached_class_B = pruning_decisions_test[i][1]
        if starting_qid_A in sequenced_decisions and reached_class_B in sequenced_decisions[starting_qid_A]:
            vec_pruned_AB = utils.pad_sequence(sequenced_decisions[starting_qid_A][reached_class_B], sequence_length, padding)
            
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
                                                          distances_hashmap,
                                                          pruning_train)

                while nb_used_analogies < nb_test_analogies:
                    if training_analogy_index == len(pruning):
                        break
                    starting_qid_C = pruning[training_analogy_index][0]
                    reached_class_D = pruning[training_analogy_index][1]
                    if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C]:
                        vec_pruned_CD = utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length, padding)
                        utils.add_sequence_analogy("test", vec_pruned_CD, vec_pruned_AB, properties, valid_analogies_pattern, "pp", test_features, test_labels)
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
                                                          distances_hashmap,
                                                          keeping_train)

                # Invalid analogies
                while nb_used_analogies < nb_test_analogies:
                    if training_analogy_index == len(keeping):
                        break
                    starting_qid_C = keeping[training_analogy_index][0]
                    reached_class_D = keeping[training_analogy_index][1]
                    if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C]:
                        vec_kept_CD = utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length, padding)
                        # Keeping decisions :: Pruning decisions
                        utils.add_sequence_analogy("test", vec_kept_CD, vec_pruned_AB, properties, valid_analogies_pattern, "kp", test_features, test_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1


def get_evaluation_analogies(decision_to_test,
                             keeping_train,
                             pruning_train, 
                             nb_keeping_in_test,
                             nb_pruning_in_test,
                             sequenced_decisions,
                             sequence_length,
                             padding,
                             properties,
                             valid_analogies_pattern,
                             invalid_analogies_pattern,
                             analogies_stats,
                             distances_hashmap,
                             knn):
    
    keeping_test = []
    pruning_test = []

    keeping_decisions = keeping_train.values.tolist()
    pruning_decisions = pruning_train.values.tolist()
    keeping_to_shuffle = keeping_decisions.copy()
    pruning_to_shuffle = pruning_decisions.copy()

    # Decision to test :: Keeping decisions
    starting_qid_A = decision_to_test[0]
    reached_class_B = decision_to_test[1]
    if starting_qid_A in sequenced_decisions and reached_class_B in sequenced_decisions[starting_qid_A]:
        vec_kept_AB = utils.pad_sequence(sequenced_decisions[starting_qid_A][reached_class_B], sequence_length, padding)
        
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
                                                      distances_hashmap,
                                                      keeping_train)

            while nb_used_analogies < nb_keeping_in_test:
                if training_analogy_index == len(keeping):
                    break
                starting_qid_C = keeping[training_analogy_index][0]
                reached_class_D = keeping[training_analogy_index][1]
                if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C]:
                    vec_kept_CD = utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length, padding)
                    
                    utils.add_evaluation_sequence_analogy(keeping_test,
                                            keeping[training_analogy_index],
                                            decision_to_test,
                                            vec_kept_CD,
                                            vec_kept_AB,
                                            analogies_stats,
                                            "keep",
                                            properties)
                    nb_used_analogies += 1
                training_analogy_index += 1
        
    # Decision to test :: Pruning decisions
    starting_qid_A = decision_to_test[0]
    reached_class_B = decision_to_test[1]
    if starting_qid_A in sequenced_decisions and reached_class_B in sequenced_decisions[starting_qid_A]:
        vec_pruned_AB = utils.pad_sequence(sequenced_decisions[starting_qid_A][reached_class_B], sequence_length, padding)
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
                                                      distances_hashmap,
                                                      pruning_train)

            while nb_used_analogies < nb_pruning_in_test:
                if training_analogy_index == len(pruning):
                    break
                starting_qid_C = pruning[training_analogy_index][0]
                reached_class_D = pruning[training_analogy_index][1]
                if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C]:
                    vec_pruned_CD = utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length, padding)
                    utils.add_evaluation_sequence_analogy(pruning_test,
                                           pruning[training_analogy_index],
                                           decision_to_test,
                                           vec_pruned_CD,
                                           vec_pruned_AB,
                                           analogies_stats,
                                           "prune",
                                           properties)
                    nb_used_analogies += 1
                training_analogy_index += 1
        
    return keeping_test, pruning_test


def main():

    parser = argparse.ArgumentParser(prog="analogy_pruning", description="Analogy-based classifier")
    parser.add_argument("--folds", dest="folds_path", help="File containing folds", required=True)
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--distances-hashmap", dest="distances_hashmap", help="Hashmap containing Euclidean distances between all pairs of Wikidata QIDS", required=True)
    parser.add_argument("--knn", nargs="+")
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--nb-training-analogies-per-decision", dest="nb_training_analogies_per_decision", 
                        required=False, default=20, type=int, help="Number of training analogies per decision")
    parser.add_argument("--nb-test-analogies", dest="nb_test_analogies", type=int, help="Number of test analogies")
    parser.add_argument("--nb-keeping-in-test", dest="nb_keeping_in_test", type=int, help="Number of keeping test analogies", default=20)
    parser.add_argument("--nb-pruning-in-test", dest="nb_pruning_in_test", type=int, help="Number of pruning test analogies", default=20)
    parser.add_argument("--sequenced-decisions", dest="sequenced_decisions", required=True, 
                        help="Pickle file containing the sequenced decisions from generate_sequenced_decisions script")
    parser.add_argument("--sequence-length", dest="sequence_length", help="Length of sequences of the decisions", default=4, type=int)
    parser.add_argument("--padding", help="Zero padding mode", default="after", choices=["before", "between", "after"])
    parser.add_argument("--analogical-properties", dest="analogical_properties", nargs='+')
    parser.add_argument("--valid-analogies-pattern", dest="valid_analogies_pattern", nargs='+')
    parser.add_argument("--invalid-analogies-pattern", dest="invalid_analogies_pattern", nargs='+')
    parser.add_argument("--nb-filters1", dest="nb_filters1", type=int)
    parser.add_argument("--nb-filters2", dest="nb_filters2", type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", help="Learning rate", required=True, type=float)
    parser.add_argument("--predictions-output", dest="predictions_output", help="Output predictions pickle hashmap", required=True)
    parser.add_argument("--stats-file", dest="stats_file", help="Output file containing statitics on analogies", required=True)
    parser.add_argument("--epochs", help="Number of epochs to train models", required=False, default=1, type=int)
    parser.add_argument("--dropout", required=True, type=float)
    args = parser.parse_args()

    start = time.time()

    # Logging parameters
    logger = logging.getLogger()
    tqdm_logging_handler = TqdmLoggingHandler()
    tqdm_logging_handler.setFormatter(logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(message)s"))
    logger.addHandler(tqdm_logging_handler)
    logger.setLevel(logging.INFO)

    # Load folds
    folds = pickle.load(open(args.folds_path, "rb"))

    # Load Wikidata hashmap (QID -> properties)
    dump = lmdb.open(args.wikidata_hashmap, readonly=True, readahead=False)
    wikidata_hashmap = dump.begin()

    # Load embeddings (QID URL -> embedding)
    embeddings = lmdb.open(args.embeddings_hashmap, readonly=True, readahead=False)
    embedding_hashmap = embeddings.begin()

    # Load distances hashmap
    distances = lmdb.open(args.distances_hashmap, readonly=True, readahead=False)
    distances_hashmap = distances.begin()

    # Loading decision file
    decisions = pandas.read_csv(args.decisions)

    # Load sequenced decisions
    sequenced_decisions = pickle.load(open(args.sequenced_decisions, "rb"))

    # Predictions for voting thresholds
    predictions = dict()

    for fold in tqdm.tqdm(folds, desc="fold"):

        predictions[fold] = dict()

        val_set = set(folds[(fold+1)%5]["test"])
        val_features = []
        val_labels = []

        # Train model
        train_set = set(folds[fold]["train"]) - val_set
        train_features = []
        train_labels = []

        logger.info("Building training data")
        keeping_decisions_train = decisions[(decisions["from"].isin(train_set)) & (decisions["target"] == 1)]
        pruning_decisions_train = decisions[(decisions["from"].isin(train_set)) & (decisions["target"] == 0)]

        # Generate analogies from keeping and pruning decisions
        get_analogies(train_features,
                      train_labels, 
                      keeping_decisions_train[["from", "QID", "depth", "starting label", "label"]], 
                      pruning_decisions_train[["from", "QID", "depth", "starting label", "label"]], 
                      args.nb_training_analogies_per_decision,
                      sequenced_decisions,
                      args.sequence_length,
                      args.padding,
                      args.analogical_properties,
                      args.valid_analogies_pattern,
                      args.invalid_analogies_pattern,
                      distances_hashmap,
                      args.knn)
        
        logger.info("Building validation analogies")
        keeping_decisions_val = decisions[(decisions["from"].isin(val_set)) & (decisions["target"] == 1)]
        pruning_decisions_val = decisions[(decisions["from"].isin(val_set)) & (decisions["target"] == 0)]

        # Generate analogies from keeping and pruning decisions
        get_testing_analogies(val_features,
                              val_labels,
                              keeping_decisions_val[["from", "QID", "depth", "starting label", "label"]].values.tolist(), 
                              pruning_decisions_val[["from", "QID", "depth", "starting label", "label"]].values.tolist(),
                              keeping_decisions_train[["from", "QID", "depth", "starting label", "label"]],
                              pruning_decisions_train[["from", "QID", "depth", "starting label", "label"]],
                              args.nb_test_analogies,
                              sequenced_decisions,
                              args.sequence_length,
                              args.padding,
                              args.analogical_properties,
                              args.valid_analogies_pattern,
                              args.invalid_analogies_pattern,
                              distances_hashmap,
                              args.knn)
        
        val_features = numpy.transpose(val_features, (0, 2, 1, 3))
        val_labels = numpy.array(val_labels)
        
        # Tranform to the input format of analogy-based convolutional model
        train_features = numpy.transpose(train_features, (0, 2, 1, 3))
        train_labels = numpy.array(train_labels)

        # Training
        sequenced_analogy_classifier = sequenced_analogy_model(args.sequence_length, args.nb_filters1, args.nb_filters2, args.dropout)
        sequenced_analogy_classifier.summary()
        sequenced_analogy_classifier.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                                metrics=[tf.keras.metrics.Precision(), 
                                         tf.keras.metrics.Recall(), 
                                         tf.keras.metrics.BinaryAccuracy()])
        
        weight_for_0 = 1 / numpy.count_nonzero(train_labels == 0) * (len(train_labels) / 2.0)
        weight_for_1 = 1 / numpy.count_nonzero(train_labels == 1) * (len(train_labels) / 2.0)

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")

        history = sequenced_analogy_classifier.fit(train_features,
                               train_labels,
                               validation_data=(val_features, val_labels),
                               callbacks=[early_stopping_cb],
                               class_weight={0: weight_for_0, 1: weight_for_1},
                               shuffle=True,
                               epochs=args.epochs)
        
        # Testing analogies then decisions
        logger.info("Testing Analogies")
        test_set = folds[fold]["test"]

        test_features = []
        test_labels = []

        logger.info("Building testing analogies")
        keeping_decisions_test = decisions[(decisions["from"].isin(test_set)) & (decisions["target"] == 1)]
        pruning_decisions_test = decisions[(decisions["from"].isin(test_set)) & (decisions["target"] == 0)]

        # Generate analogies from keeping and pruning decisions
        get_testing_analogies(test_features,
                              test_labels,
                              keeping_decisions_test[["from", "QID", "depth", "starting label", "label"]].values.tolist(), 
                              pruning_decisions_test[["from", "QID", "depth", "starting label", "label"]].values.tolist(),
                              keeping_decisions_train[["from", "QID", "depth", "starting label", "label"]],
                              pruning_decisions_train[["from", "QID", "depth", "starting label", "label"]],
                              args.nb_test_analogies,
                              sequenced_decisions,
                              args.sequence_length,
                              args.padding,
                              args.analogical_properties,
                              args.valid_analogies_pattern,
                              args.invalid_analogies_pattern,
                              distances_hashmap,
                              args.knn)
        
        test_features = numpy.transpose(test_features, (0, 2, 1, 3))
        test_labels = numpy.array(test_labels)

        # Analogies evaluation
        sequenced_analogy_classifier.evaluate(test_features, test_labels)
        
        # Statistics for analogies
        analogies_stats = AnalogyStatistics.AnalogyStats()
        logger.info("Expansion --> evaluation of decisions")
        # Start the expansion
        for qid in tqdm.tqdm(test_set):
            depth = 0
            qid_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + qid + ">", embedding_hashmap)

            if qid_emb is not None:

                predictions[fold][qid] = dict()

                q_adjacency = utils.get_hashmap_content(qid, wikidata_hashmap)
                seen_classes = {qid}
                reached_q_w_decisions = set(decisions[(decisions["from"] == qid)]["QID"])

                if q_adjacency is not None and ("P31" in q_adjacency["claims"] or "P279" in q_adjacency["claims"] or
                                "(-)P279" in q_adjacency["claims"]):
                    
                    classes = set()

                    if "P31" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["P31"]) & reached_q_w_decisions
                    if "P279" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["P279"]) & reached_q_w_decisions
                    if "(-)P279" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["(-)P279"]) & reached_q_w_decisions

                    while classes:
                        seen_classes |= classes
                        new_classes = set()
                        depth += 1

                        for cl in classes:
                            cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)
                            if cl_emb is not None:

                                predictions[fold][qid][cl] = dict()

                                cl_decision = decisions[(decisions["from"] == qid) & (decisions["QID"] == cl)]

                                # Generate analogies
                                keeping_test_analogies, pruning_test_analogies = get_evaluation_analogies([qid, cl, depth, cl_decision["starting label"].iloc[0], cl_decision["label"].iloc[0]],
                                                                                                        keeping_decisions_train[["from", "QID", "depth", "starting label", "label"]],
                                                                                                        pruning_decisions_train[["from", "QID", "depth", "starting label", "label"]],
                                                                                                        args.nb_keeping_in_test,
                                                                                                        args.nb_pruning_in_test,
                                                                                                        sequenced_decisions,
                                                                                                        args.sequence_length,
                                                                                                        args.padding,
                                                                                                        args.analogical_properties,
                                                                                                        args.valid_analogies_pattern,
                                                                                                        args.invalid_analogies_pattern,
                                                                                                        analogies_stats,
                                                                                                        distances_hashmap,
                                                                                                        args.knn)

                                if len(keeping_test_analogies) > 0:
                                    # Transform for input format of the model
                                    keeping_test_analogies = numpy.transpose(keeping_test_analogies, (0, 2, 1, 3))
                                    # Output (predictions on analogies)
                                    y_keeping_pred = 0
                                    if args.dropout > 0:
                                        probas = numpy.stack([sequenced_analogy_classifier(keeping_test_analogies, training=True) for sample in range(10)])
                                        y_keeping_pred = probas.mean(axis=0)
                                    else:
                                        y_keeping_pred = sequenced_analogy_classifier.predict(keeping_test_analogies, verbose=0)
                                    predictions[fold][qid][cl]["keeping"] = y_keeping_pred
                                    
                                if len(pruning_test_analogies) > 0:
                                    # Transform for input format of the model
                                    pruning_test_analogies = numpy.transpose(pruning_test_analogies, (0, 2, 1, 3))
                                    # Output (predictions on analogies)
                                    y_pruning_pred = 0
                                    if args.dropout > 0:
                                        probas = numpy.stack([sequenced_analogy_classifier(pruning_test_analogies, training=True) for sample in range(10)])
                                        y_pruning_pred = probas.mean(axis=0)
                                    else:
                                        y_pruning_pred = sequenced_analogy_classifier.predict(pruning_test_analogies, verbose=0)
                                    predictions[fold][qid][cl]["pruning"] = y_pruning_pred
                                
                                cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)
                                if cl_adjacency is not None:
                                    new_classes |= utils.get_unseen_subclasses(cl_adjacency, seen_classes) & reached_q_w_decisions

                        classes = new_classes

    pickle.dump(predictions, open(args.predictions_output, "wb"))
    logger.info(f"Execution time = {utils.convert(time.time() - start)}")

if __name__ == '__main__':
    main()