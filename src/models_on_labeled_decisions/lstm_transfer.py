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

import numpy
import pandas
import lmdb
import tqdm
import tensorflow as tf

import utils
from TqdmLoggingHandler import *


def lstm_model(units=50, input_shape=8):
    """ Build LSTM model based on paths in the graph (sequences of the model). """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=units, input_shape=(input_shape, 200), return_sequences=False, 
                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                             bias_regularizer=tf.keras.regularizers.L2(1e-4),
                             activity_regularizer=tf.keras.regularizers.L2(1e-5)),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    return model


def pad_lstm_sequence(seq, max_length, pad_mode="after"):
    """ zero padding after QID and classes by default """
    sequence = seq
    pad = max_length - len(seq)
    if pad >= 0:
        if pad_mode == "after":
            sequence = sequence + pad * [numpy.zeros(200)]
        elif pad_mode == "before":
            sequence = pad * [numpy.zeros(200)] + sequence
        else:
            sequence = [sequence[0]] + pad * [numpy.zeros(200)] + sequence[1:]
    else:
        sequence = [sequence[0]] + sequence[-pad+1:]
        
    return sequence


def main():

    parser = argparse.ArgumentParser(prog="lstm_pruning", description="Long-Short-Term-Memory Classifier")
    parser.add_argument("--padding", help="Zero padding method for lstm and sequenced analogy", 
                        choices=['before', 'between', 'after'], required=False, default="after")
    parser.add_argument("--sequence-len", dest="sequence_len", help="Length of the sequences", default=8, type=int)
    parser.add_argument("--folds", dest="folds_path", help="File containing folds", required=True)
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--predictions-output", dest="predictions_output", help="Output predictions pickle hashmap", required=True)
    parser.add_argument("--nb-units", dest="nb_units", help="Number of units in LSTM model", required=False, type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", help="Learning rate", required=True, type=float)
    parser.add_argument("--epochs", help="Number of epochs to train models", required=False, default=200, type=int)
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

    # Loading decision file
    decisions = pandas.read_csv(args.decisions)

    # Predictions for voting thresholds
    predictions = dict()

    for fold in tqdm.tqdm(folds, desc="fold"):

        predictions[fold] = dict()

        val_set = set(folds[fold]["train"][:int(len(folds[fold]["train"])/5)])
        val_features = []
        val_labels = []

        # Train model
        train_set = set(folds[fold]["train"]) - val_set
        train_features = []
        train_labels = []

        logger.info("Building training data")
        for qid in tqdm.tqdm(train_set):

            qid_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + qid + ">", embedding_hashmap)

            if qid_emb is not None:

                sequences = {}

                q_adjacency = utils.get_hashmap_content(qid, wikidata_hashmap)
                seen_classes = {qid}

                if q_adjacency is not None and "P31" in q_adjacency["claims"]:
                    reached_q_w_decisions = set(decisions[(decisions["from"] == qid)]["QID"])
                    classes = utils.get_relation_objects(q_adjacency["claims"]["P31"]) & reached_q_w_decisions

                    for cl in classes:
                        cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)
                        if cl_emb is not None:
                            sequence = [qid_emb, cl_emb]
                            sequences[cl] = sequence
                            train_features.append(pad_lstm_sequence(sequence, args.sequence_len, args.padding))
                            train_labels.append(decisions[(decisions["from"] == qid) & (decisions["QID"] == cl)]["target"].iloc[0])

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
                                        sequence = sequences[cl] + [sub_emb]
                                        sequences[sub_cl] = sequence
                                        train_features.append(pad_lstm_sequence(sequence, args.sequence_len, args.padding))
                                        train_labels.append(decisions[(decisions["from"] == qid) & (decisions["QID"] == sub_cl)]["target"].iloc[0])
                                new_classes |= sub_classes

                        classes = new_classes

        logger.info("Building validation data")
        for qid in tqdm.tqdm(val_set):

            qid_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + qid + ">", embedding_hashmap)

            if qid_emb is not None:

                sequences = {}

                q_adjacency = utils.get_hashmap_content(qid, wikidata_hashmap)
                seen_classes = {qid}

                if q_adjacency is not None and "P31" in q_adjacency["claims"]:
                    reached_q_w_decisions = set(decisions[(decisions["from"] == qid)]["QID"])
                    classes = utils.get_relation_objects(q_adjacency["claims"]["P31"]) & reached_q_w_decisions

                    for cl in classes:
                        cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)
                        if cl_emb is not None:
                            sequence = [qid_emb, cl_emb]
                            sequences[cl] = sequence
                            val_features.append(pad_lstm_sequence(sequence, args.sequence_len, args.padding))
                            val_labels.append(decisions[(decisions["from"] == qid) & (decisions["QID"] == cl)]["target"].iloc[0])

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
                                        sequence = sequences[cl] + [sub_emb]
                                        sequences[sub_cl] = sequence
                                        val_features.append(pad_lstm_sequence(sequence, args.sequence_len, args.padding))
                                        val_labels.append(decisions[(decisions["from"] == qid) & (decisions["QID"] == sub_cl)]["target"].iloc[0])
                                new_classes |= sub_classes

                        classes = new_classes

        train_features = numpy.array(train_features)
        train_labels = numpy.array(train_labels)

        val_features = numpy.array(val_features)
        val_labels = numpy.array(val_labels)

        weight_for_0 = 1 / numpy.count_nonzero(train_labels == 0) * (len(train_labels) / 2.0)
        weight_for_1 = 1 / numpy.count_nonzero(train_labels == 1) * (len(train_labels) / 2.0)

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss")

        lstm = lstm_model(units=args.nb_units, input_shape=args.sequence_len)
        lstm.summary()
        lstm.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                                metrics=[tf.keras.metrics.Precision(), 
                                         tf.keras.metrics.Recall(), 
                                         tf.keras.metrics.BinaryAccuracy()])
        lstm.fit(train_features,
                 train_labels,
                 validation_data=(val_features, val_labels),
                 epochs=args.epochs, 
                 shuffle=True, 
                 class_weight={0: weight_for_0, 1: weight_for_1},
                 callbacks=[early_stopping_cb])

        # Test model
        logger.info("Testing LSTM")
        test_set = folds[fold]["test"]

        for qid in tqdm.tqdm(test_set, desc="test"):
            depth = 0
            qid_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + qid + ">", embedding_hashmap)

            if qid_emb is not None:

                predictions[fold][qid] = dict()

                sequences = {}

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

                    depth += 1

                    for cl in classes:
                        cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)
                        if cl_emb is not None:
                            sequence = [qid_emb, cl_emb]
                            sequences[cl] = sequence
                            predictions[fold][qid][cl] = lstm.predict(numpy.array([pad_lstm_sequence(sequence, args.sequence_len, args.padding)]),verbose=0)[0]
                    # Start expansion
                    while classes:
                        seen_classes |= classes
                        new_classes = set()
                        depth += 1

                        # Prune or keep classes to traverse
                        for cl in classes:
                            cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)
                            cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)

                            if cl_emb is not None:
                                if cl_adjacency is not None:
                                    sub_classes = utils.get_unseen_subclasses(cl_adjacency, seen_classes) & reached_q_w_decisions
                                    for sub_cl in sub_classes:
                                        sub_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + sub_cl + '>', embedding_hashmap)
                                        if sub_emb is not None:
                                            sequence = sequences[cl] + [sub_emb]
                                            sequences[sub_cl] = sequence
                                            predictions[fold][qid][sub_cl] = lstm.predict(numpy.array([pad_lstm_sequence(sequence, args.sequence_len, args.padding)]), verbose=0)[0]
                                            new_classes.add(sub_cl)

                        classes = new_classes

    pickle.dump(predictions, open(args.predictions_output, "wb"))
    logger.info(f"Execution time = {utils.convert(time.time() - start)}")


if __name__ == '__main__':
    main()
