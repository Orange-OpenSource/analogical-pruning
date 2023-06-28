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
import tensorflow as tf

import utils
from TqdmLoggingHandler import *


def mlp_model(hidden_layers, dropout):
    """ Build MLP model """
    model = tf.keras.Sequential()
    for layer in hidden_layers:
        model.add(tf.keras.layers.Dense(layer, activation="relu"))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return model


def main():
    start = time.time()

    parser = argparse.ArgumentParser(prog="mlp_pruning", description="Multi-layer Perceptron classifier")
    parser.add_argument("--folds", dest="folds_path", help="File containing folds", required=True)
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--wikidata", dest="wikidata_hashmap", help="Folder containing the LMDB Wikidata hashmap",
                        required=True)
    parser.add_argument("--embeddings", dest="embeddings_hashmap", help="Folder containing the LMDB embeddings hashmap",
                        required=True)
    parser.add_argument("--concatenation", dest="concatenation", help="Type of embedding concatenation",
                        choices=["translation", "horizontal"], required=True)
    parser.add_argument("--predictions-output", dest="predictions_output", help="Output predictions pickle hashmap", required=True)
    parser.add_argument("--hidden-layers", dest="hidden_layers", help="Hidden layers", required=True, nargs="+", type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", help="Learning rate", required=True, type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--epochs", help="Number of epochs to train models", required=False, default=10, type=int)

    args = parser.parse_args()

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

        val_set = set(folds[(fold+1)%5]["test"])
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
                for _, r in decisions[(decisions["from"] == qid)].iterrows():
                    qid2_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + r["QID"] + ">", embedding_hashmap)

                    if qid2_emb is not None:
                        train_labels.append(r["target"])  # 1 : Keep, 0 : Prune

                        if args.concatenation == "translation":
                            train_features.append(qid2_emb - qid_emb)
                        else:
                            train_features.append(numpy.hstack((qid_emb, qid2_emb)))
        
        logger.info("Building validation data")
        for qid in tqdm.tqdm(val_set):
            qid_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + qid + ">", embedding_hashmap)

            if qid_emb is not None:
                for _, r in decisions[(decisions["from"] == qid)].iterrows():
                    qid2_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + r["QID"] + ">", embedding_hashmap)

                    if qid2_emb is not None:
                        val_labels.append(r["target"])  # 1 : Keep, 0 : Prune

                        if args.concatenation == "translation":
                            val_features.append(qid2_emb - qid_emb)
                        else:
                            val_features.append(numpy.hstack((qid_emb, qid2_emb)))
        
        train_features = numpy.array(train_features)
        train_labels = numpy.array(train_labels)

        val_features = numpy.array(val_features)
        val_labels = numpy.array(val_labels)

        weight_for_0 = 1 / numpy.count_nonzero(train_labels == 0) * (len(train_labels) / 2.0)
        weight_for_1 = 1 / numpy.count_nonzero(train_labels == 1) * (len(train_labels) / 2.0)

        logger.info("Training MLP")
        mlp = mlp_model(args.hidden_layers, args.dropout)
        mlp.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                                metrics=[tf.keras.metrics.Precision(), 
                                         tf.keras.metrics.Recall(), 
                                         tf.keras.metrics.BinaryAccuracy()])
        
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss")

        history = mlp.fit(train_features, 
                                train_labels, 
                                epochs=args.epochs, 
                                shuffle=True,
                                validation_data=(val_features, val_labels),
                                callbacks=[early_stopping_cb],
                                class_weight={0: weight_for_0, 
                                              1: weight_for_1})

        # Test model
        logger.info("Testing MLP")
        test_set = folds[fold]["test"]

        for q in tqdm.tqdm(test_set, desc="test"):
            q_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + q + ">", embedding_hashmap)
            if q_emb is not None:
                reached_q_w_decisions = set(decisions[(decisions["from"] == q)]["QID"])
                q_adjacency = utils.get_hashmap_content(q, wikidata_hashmap)
                seen_classes = {q}
                depth = 0

                predictions[fold][q] = dict()

                if q_adjacency is not None and ("P31" in q_adjacency["claims"] or "P279" in q_adjacency["claims"] or
                                            "(-)P279" in q_adjacency["claims"]):
                    
                    classes = set()

                    if "P31" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["P31"]) & reached_q_w_decisions
                    if "P279" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["P279"]) & reached_q_w_decisions
                    if "(-)P279" in q_adjacency["claims"]:
                        classes |= utils.get_relation_objects(q_adjacency["claims"]["(-)P279"]) & reached_q_w_decisions 

                    # Start expansion
                    while classes:
                        depth += 1
                        seen_classes |= classes
                        new_classes = set()

                        # Prune or keep classes to traverse
                        for cl in classes:
                            cl_adjacency = utils.get_hashmap_content(cl, wikidata_hashmap)
                            cl_emb = utils.get_hashmap_content(utils.WIKIDATA_PREFIX + cl + '>', embedding_hashmap)

                            if cl_emb is not None:
                                cl_feature = None
                                if args.concatenation == "translation":
                                    cl_feature = cl_emb - q_emb
                                else:
                                    cl_feature = numpy.hstack((q_emb, cl_emb))

                                predictions[fold][q][cl] = mlp.predict(numpy.array([cl_feature]), verbose=0)[0]
                                if cl_adjacency is not None:
                                    new_classes |= (utils.get_unseen_subclasses(cl_adjacency, seen_classes) & reached_q_w_decisions)

                        classes = new_classes

    pickle.dump(predictions, open(args.predictions_output, "wb"))
    logger.info(f"Execution time = {utils.convert(time.time() - start)}")


if __name__ == '__main__':
    main()
