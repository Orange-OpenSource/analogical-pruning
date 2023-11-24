# Knowledge graph bootstrapping with Wikidata: subgraph selection with zero-shot analogical pruning

These scripts provide several analogy-based (and other) classifiers to select subgraphs of interest from Wikidata to bootstrap a Knowledge Graph.
Specifically, given a set of seed QIDs of interest, a graph expansion is performed following P31, P279, and (-)P279 edges.
To select thematic subgraphs of interest, traversed classes that thematically deviates from seed QIDs of interest are pruned by several proposed and compared classifiers relying on learned graph embedings:
* Analogy-based classifiers (with or without sequences)
* Random forest
* MLP
* LSTM
* SVM
* Depth-based pruning
* Threshold-based pruning (node degree, distance in the embedding space) [1]

## Repository structure

* ``data/``: contains datasets (see [Available datasets](#available-datasets)) and should contain the necessary hashmaps (see [Data requirements](#data-requirements))
* ``src/models_no_constraints``: contains python scripts to execute pruning models. These models take as input a set of seed QIDs and, for supervised models, a set of labeled decisions to train. Then, these models perform an expansion along the ontology hierarchy of Wikidata from the seed QIDs and stop when they decide to prune. They output a number of reached classes.
* ``src/models_on_labeled_decisions``: contains python scripts to execute pruning models. These models take as input a set labeled decisions in folds to train and test. These models perform an expansion along the ontology hierarchy of Wikidata from the starting QIDs of the dataset and limited to the labeled classes in the dataset. They output a pickle file containing the models' results.
* ``src/utils``: contains utility python scripts

## Requirements

### Python requirements 

See [requirements.txt](requirements.txt)

* Python 3.11
* numpy == 1.23.5
* tensorflow == 2.12.0rc1
* pandas == 1.5.3
* matplotlib == 3.7.1
* lmdb == 1.4.0
* scikit-learn == 1.2.1
* tqdm == 4.65.0

### Data requirements

Some scripts require:
* A Wikidata hashmap: a LMDB hashmap structured as follows
```
QID -> pickle object(
  dict(
    "labels": dict(
        "en": list(QID labels), 
        "fr": list(QID labels)
    ),
    "claims": dict(
      "P31": list(dict(
          "type": "item",
          "value": "Linked QID",
          "rank": "rank of the claim in Wikidata"
        ), ...
      ),
      "P279": list(...),
      "(-)P279: list(...),
    ) 
  )
)
```
* An embedding hashmap: a LMDB hashmap structured as follows
```
<http://www.wikidata.org/entity/QID> -> pickle object(
    numpy.array([embedding vector of QID])
)
```

We have two such hashmaps:
* One that associates each QID with its [pre-trained embeddings from Pytorch Biggraph](https://torchbiggraph.readthedocs.io/en/latest/pretrained_embeddings.html)
* One that associates each QID with the centroid of the embeddings of its instances from the [pre-trained embeddings from Pytorch Biggraph](https://torchbiggraph.readthedocs.io/en/latest/pretrained_embeddings.html) 

## Available datasets

Datasets are also available on [Zenodo](https://doi.org/10.5281/zenodo.8091584).

| Dataset                   | # Seed QIDs | # Labeled decisions | # Prune decisions | Min prune depth | Max prune depth | # Keep decisions | Min keep depth | Max keep depth | # Reached nodes up | # Reached nodes down |
|---------------------------|-------------|---------------------|-------------------|-----------------|-----------------|------------------|----------------|----------------|--------------------|----------------------|
| [dataset1](data/dataset1) | 455         | 5233                | 3464              | 1               | 4               | 1769             | 1              | 4              | 1507               | 2593609              |
| [dataset2](data/dataset2) | 105         | 982                 | 388               | 1               | 2               | 594              | 1              | 3              | 1159               | 1247385              |


Each dataset folder contains
* ``datasetX.csv``: a CSV file containing one seed QID per line (not the complete URL, just the QID). This CSV file has no header. 
* ``datasetX_labels.csv``: a CSV file containing one seed QID per line and its label (not the complete URL, just the QID)
* ``datasetX_gold_decisions.csv``: a CSV file with seed QIDs, reached QIDs, and the labeled decision (1: keep, 0: prune)
* ``datasetX_Y_folds.pkl``: folds to train and test models based on the labeled decisions

## Available scripts

### In ``models_no_constraints`` folder

#### ``analogy_pruning.py``

Performs the same expansion as ``down_expansion.py`` from a set of seed QIDs but prunes sub-classes with a analogy-based convolutional model.

```
python analogy_pruning.py --wikidata WIKIDATA --seed-qids QIDS --embeddings EMBEEDINGS --decisions DECISIONS --epochs EPOCHS --output-statistics OUTPUT --relevant-entities RELEVANT_ENTITIES --voting VOTING --voting-threshold THRESHOLD --knn KNN --nb-training-analogies-per-decision TRAINING_ANALOGIES --nb-test-analogies TEST_ANALOGIES --nb-keeping-in-test KEEPING_ANALOGIES --nb-pruning-in-test PRUNING_ANALOGIES --analogical-properties PROPERTIES --valid-analogies-pattern VALID_PATTERN --invalid-analogies-pattern INVALID_PATTERN --node-degree NODE_DEGREE --analogy-batch ANALOGY_BATCH --dropout DROPOUT --nb-filters1 FILTERS1 --nb-filters2 FILTERS2 --learning-rate LEARNING_RATE
```

with
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path of the CSV file where statistics of the down expansion will be stored
* ``RELEVANT_ENTITIES``: path of the JSON file where relevant entities will be stored ({"QID": ["Q1", "Q2", ...], ...})
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``EPOCHS``: number of epochs to train the model
* ``VOTING``: strategy of voting (majority or weighted)
* ``THRESHOLD``: value in [0, 1] that takes the decision if an analogy is either valid or invalid
* ``KNN``: apply K-Nearest-Neighbors on the train set, the test set, both or none
* ``TRAINING_ANALOGIES``: number of training analogies per decision
* ``TEST_ANALOGIERS``: number of testing analogies per decision
* ``KEEPING_ANALOGIES``: number of keeping analogies in test set to evaluate the model on the expansion
* ``PRUNING_ANALOGIES``: number of pruning analogies in test set to evaluate the model on the expansion
* ``PROPERTIES``: properties to consider to build the analogies (training and test set) among relfexivity, inner-symmetry, or symmetry
* ``VALID_PATTERN``: valid pattern of analogies
* ``INVALID_PATTERN``: invalid pattern of analogies
* ``NODE_DEGREE``: node degree to limit the selection of entities from classes not exceeding that degree
* ``ANALOGY_BATCH``: batch size to speed up predictions
* ``DROPOUT``: rate of neurons dropout after each convolutional layer
* ``FILTERS1``: number of filters in first convolutional layer
* ``FILTERS2``: number of filters in second convolutional layer
* ``LEARNING_RATE``: learning rate

#### ``depth_pruning.py``

Performs a downward expansion from a set of seed QIDs limited by a depth threshold. 
Specifically, starting from seed QIDs, their direct classes will be retrieved through 1 hop P31, P279 and (-)P279 edges. 
From them, only (-)P279 edges will be followed when expanding until reaching the depth threshold.

```
python depth_pruning.py --depth-threshold DEPTH --wikidata WIKIDATA --seed-qids QIDS --output OUTPUT
```

with
* ``DEPTH``: the maximal depth that can be reached. Direct classes are considered at depth 1
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path of the CSV file where statistics of the down expansion will be stored

#### ``down_expansion.py``

Performs a downward expansion from a set of seed QIDs. 
Specifically, starting from seed QIDs, their direct classes will be retrieved through 1 hop P31, P279 and (-)P279 edges. 
From them, only (-)P279 edges will be followed when expanding until reaching leaves.

```
python down_expansion.py --wikidata WIKIDATA --seed-qids QIDS --output OUTPUT
```

with
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path of the CSV file where statistics of the down expansion will be stored

#### ``lstm_pruning.py``

Performs the same expansion as ``down_expansion.py`` from a set of seed QIDs but prunes sub-classes based on LSTM model which is trained on a set of labeled decisions. The LSTM model takes as input the concatenation of the embeddings of the path leading to the sub-class with zero padding.

```
python lstm_pruning.py --wikidata WIKIDATA --seed-qids QIDS --embeddings EMBEDDINGS --decisions DECISIONS --epochs EPOCHS --sequence-len LENGTH --padding PADDING --nb-units UNITS --learning-rate LEARNING_RATE --output-statistics OUTPUT
```

with
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path of the CSV file where statistics of the down expansion will be stored
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``EPOCHS``: number of epochs to train the model
* ``LENGTH``: length of the path to consider for pruning
* ``PADDING``: type of zero padding (before, between, or after the path between seed QID and sub-class)
* ``UNITS``: number of units in LTSM
* ``LEARNING_RATE``: learning rate

#### ``mlp_pruning.py``

Performs the same expansion as ``down_expansion.py`` from a set of seed QIDs but prunes sub-classes based on MLP model which is trained on a set of labeled decisions. The MLP model takes as input either the concatenation of the embedding of the seed QID and the embedding of the sub-class, or the translation between these two embeddings.

```
python mlp_pruning.py --wikidata WIKIDATA --seed-qids QIDS --embeddings EMBEDDINGS --decisions DECISIONS --epochs EPOCHS --concatenation CONCATENATION --hidden-layers LAYERS --learning-rate LEARNING_RATE --dropout DROPOUT --output-statistics OUTPUT
```

with
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path of the CSV file where statistics of the down expansion will be stored
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``EPOCHS``: number of epochs to train the model
* ``CONCATENATION``: type of embeddings concatenation (translation or horizontal)
* ``LAYERS``: number of hidden layers and number of neurons of each layer
* ``LEARNING_RATE``: learning rate
* ``DROPOUT``: rate of neurons dropout after each hidden layer

#### ``random_forest_pruning.py``

Performs the same expansion as ``down_expansion.py`` from a set of seed QIDs but prunes sub-classes based on MLP model which is trained on a set of labeled decisions. The MLP model takes as input either the concatenation of the embedding of the seed QID and the embedding of the sub-class, or the translation between these two embeddings.

```
python random_forest_pruning.py --wikidata WIKIDATA --seed-qids QIDS --embeddings EMBEDDINGS --decisions DECISIONS --epochs EPOCHS --concatenation CONCATENATION --nb-estimators ESTIMATORS --output-statistics OUTPUT
```

with
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path of the CSV file where statistics of the down expansion will be stored
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``EPOCHS``: number of epochs to train the model
* ``CONCATENATION``: type of embeddings concatenation (translation or horizontal)
* ``ESTIMATORS``: number of estimators for the Random Forest

#### ``sequence_analogy_pruning.py``

Performs the same expansion as ``down_expansion.py`` from a set of seed QIDs but prunes sub-classes with a analogy-based convolutional model.

```
python sequence_analogy_pruning.py --wikidata WIKIDATA --seed-qids QIDS --embeddings EMBEEDINGS --decisions DECISIONS --epochs EPOCHS --output-statistics OUTPUT --relevant-entities RELEVANT_ENTITIES --voting VOTING --voting-threshold THRESHOLD --knn KNN --nb-training-analogies-per-decision TRAINING_ANALOGIES --nb-test-analogies TEST_ANALOGIES --nb-keeping-in-test KEEPING_ANALOGIES --nb-pruning-in-test PRUNING_ANALOGIES --sequenced-decisions SEQUENCED_DECISIONS --sequence-length SEQ_LENGTH --padding PADDING --analogical-properties PROPERTIES --valid-analogies-pattern VALID_PATTERN --invalid-analogies-pattern INVALID_PATTERN --node-degree NODE_DEGREE --analogy-batch ANALOGY_BATCH --dropout DROPOUT --nb-filters1 FILTERS1 --nb-filters2 FILTERS2 --learning-rate LEARNING_RATE
```

with
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path of the CSV file where statistics of the down expansion will be stored
* ``RELEVANT_ENTITIES``: path of the JSON file where relevant entities will be stored ({"QID": ["Q1", "Q2", ...], ...})
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``EPOCHS``: number of epochs to train the model
* ``VOTING``: strategy of voting (majority or weighted)
* ``THRESHOLD``: value in [0, 1] that takes the decision if an analogy is either valid or invalid
* ``KNN``: apply K-Nearest-Neighbors on the train set, the test set, both or none
* ``TRAINING_ANALOGIES``: number of training analogies per decision
* ``TEST_ANALOGIERS``: number of testing analogies per decision
* ``KEEPING_ANALOGIES``: number of keeping analogies in test set to evaluate the model on the expansion
* ``PRUNING_ANALOGIES``: number of pruning analogies in test set to evaluate the model on the expansion
* ``SEQUENCED_DECISIONS``: a pickle dictionary containing the path embeddings of each pair of seed QID and reached class
* ``SEQ_LENGTH``: length of the sequences to consider
* ``PADDING``: padding mode (before, between, or after)
* ``PROPERTIES``: properties to consider to build the analogies (training and test set) among relfexivity, inner-symmetry, or symmetry
* ``VALID_PATTERN``: valid pattern of analogies
* ``INVALID_PATTERN``: invalid pattern of analogies
* ``NODE_DEGREE``: node degree to limit the selection of entities from classes not exceeding that degree
* ``ANALOGY_BATCH``: batch size to speed up predictions
* ``DROPOUT``: rate of neurons dropout after each convolutional layer
* ``FILTERS1``: number of filters in first convolutional layer
* ``FILTERS2``: number of filters in second convolutional layer
* ``LEARNING_RATE``: learning rate

#### ``svm_pruning.py``

Performs the same expansion as ``down_expansion.py`` from a set of seed QIDs but prunes sub-classes based on MLP model which is trained on a set of labeled decisions. The MLP model takes as input either the concatenation of the embedding of the seed QID and the embedidng of the sub-class, or the translation between these two embeddings.

```
python svm_pruning.py --wikidata WIKIDATA --seed-qids QIDS --embeddings EMBEDDINGS --decisions DECISIONS --concatenation CONCATENATION --output-statistics OUTPUT
```

with
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path of the CSV file where statistics of the down expansion will be stored
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``CONCATENATION``: type of embeddings concatenation (translation or horizontal)

#### ``threshold_pruning.py``

Performs a downward expansion and pruning from a set of seed QIDs. 
Specifically, starting from seed QIDs, their direct classes will be retrieved through 1 hop P31 and P279 edges. 
From them, only P279 edges will be followed when expanding until reaching leaves.
Classes will be pruned according to absolute node degree threshold, relative node degree threshold, and 
relative distance in the embedding space.

```
python threshold_pruning --nd-threshold ND-THRESHOLD --alpha ALPHA --gamma GAMMA --beta BETA --wikidata WIKIDATA --embeddings EMBEDDINGS --seed-qids QIDS --output OUTPUT --output-statistics STATISTICS
```

with
* ``ND-THRESHOLD``: absolute degree threshold
* ``ALPHA``: coefficient controlling the relative degree threshold
* ``GAMMA``: minimum degree at an expansion level to enable the relative degree threshold
* ``BETA``: coefficient controlling the relative distance threshold
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``EMBEDDINGS``: path to the folder in which the embedding LMDB hashmap is stored
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``OUTPUT``: path of the CSV file where statistics of the up expansion will be stored
* ``STATISTICS``: path of the CSV file where statistics of the up expansion will be stored

#### ``up_expansion.py``

Performs an upward expansion from a set of seed QIDs. 
Specifically, starting from seed QIDs, their direct classes will be retrieved through 1 hop P31 and P279 edges. 
From them, only P279 edges will be followed when expanding until reaching leaves.

```
python down_expansion.py --wikidata WIKIDATA --seed-qids QIDS --output OUTPUT
```

with
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path of the CSV file where statistics of the up expansion will be stored

### In ``models_on_labeled_decisions`` folder

#### ``analogy_pruning.py`` and ``analogy_transfer.py ``

Performs the same expansion as ``down_expansion.py`` from a set of seed QIDs but prunes sub-classes with a analogy-based convolutional model.
The script ``analogy_transfer.py`` is intended for use in a transfer setting.

```
python analogy_pruning.py --wikidata WIKIDATA --embeddings EMBEDDINGS --folds FOLDS --decisions DECISIONS --epochs EPOCHS --distances-hashmap DISTANCES --knn KNN --nb-training-analogies-per-decision TRAINING_ANALOGIES --nb-test-analogies TEST_ANALOGIES --nb-keeping-in-test KEEPING_ANALOGIES --nb-pruning-in-test PRUNING_ANALOGIES --analogical-properties PROPERTIES --valid-analogies-pattern VALID_PATTERN --invalid-analogies-pattern INVALID_PATTERN --dropout DROPOUT --nb-filters1 FILTERS1 --nb-filters2 FILTERS2 --learning-rate LEARNING_RATE --predictions-output PREDICTIONS
```

with
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``FOLDS``: a pickle dictionary containing the folds
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``EPOCHS``: number of epochs to train the model
* ``DISTANCES``: path to the folder in which the pairwise distances between seed QIDs LMDB hashmap is stored
* ``KNN``: apply K-Nearest-Neighbors on the train set, the test set, both or none
* ``TRAINING_ANALOGIES``: number of training analogies per decision
* ``TEST_ANALOGIERS``: number of testing analogies per decision
* ``KEEPING_ANALOGIES``: number of keeping analogies in test set to evaluate the model on the expansion
* ``PRUNING_ANALOGIES``: number of pruning analogies in test set to evaluate the model on the expansion
* ``PROPERTIES``: properties to consider to build the analogies (training and test set) among relfexivity, inner-symmetry, or symmetry
* ``VALID_PATTERN``: valid pattern of analogies
* ``INVALID_PATTERN``: invalid pattern of analogies
* ``DROPOUT``: rate of neurons dropout after each convolutional layer
* ``FILTERS1``: number of filters in first convolutional layer
* ``FILTERS2``: number of filters in second convolutional layer
* ``LEARNING_RATE``: learning rate of the convolutional model
* ``PREDICTIONS``: a pickle dictionary where the output values of the model are saved and on which a vote will be applied

#### ``analogy_voting.py``

Applies a majority or weighted vote on the predictions of the analogy-based models- (with or without sequences).

```
python analogy_voting.py --wikidata WIKIDATA --embeddings EMBEDDINGS --folds FOLDS --decisions DECISIONS --predictions PREDICTIONS --voting VOTING --voting-threshold THRESHOLD --valid-analogies-pattern VALID_PATTERN --invalid-analogies-pattern INVALID_PATTERN --output OUTPUT
```

with
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``FOLDS``: a pickle dictionary containing the folds
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``PREDICTIONS``: a pickle dictionary where the output values of the model are saved and on which a vote will be applied
* ``VOTING``: strategy of voting (majority or weighted)
* ``THRESHOLD``: value in [0, 1] that takes the decision if an analogy is either valid or invalid
* ``VALID_PATTERN``: valid pattern of analogies
* ``INVALID_PATTERN``: invalid pattern of analogies
* ``OUTPUT``: path of the pickle output file where results will be stored

#### ``depth_pruning.py``

Performs a downward expansion from a set of seed QIDs limited by a depth threshold and the labeled decisions. 
Specifically, starting from seed QIDs, their direct classes will be retrieved through 1 hop P31, P279 and (-)P279 edges. 
From them, only (-)P279 edges will be followed when expanding until reaching the depth threshold (only on labeled decisions).

```
python depth_pruning.py --depth-threshold DEPTH --wikidata WIKIDATA --folds FOLDS --decisions DECISIONS --output OUTPUT
```

with
* ``DEPTH``: the maximal depth that can be reached. Direct classes are considered at depth 1
* ``FOLDS``: a pickle dictionary containing the folds to test on (only the train set will be used)
* ``DECISIONS``: CSV file containing labeled decisions
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path of the pickle output file where results will be stored

#### ``lstm_pruning.py and lstm_transfer.py``

Performs the same expansion as ``down_expansion.py`` from a set of seed QIDs but prunes sub-classes based on LSTM model which is trained on a set of labeled decisions. The LSTM model takes as input the concatenation of the embeddings of the path leading to the sub-class with zero padding.
The script ``lstm_transfer.py`` is intended for use in a transfer setting.

```
python lstm_pruning.py --wikidata WIKIDATA --embeddings EMBEDDINGS --folds FOLDS --decisions DECISIONS --epochs EPOCHS --sequence-len LENGTH --padding PADDING --nb-units UNITS --learning-rate LEARNING_RATE --predictions-output PREDICTIONS
```

with
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``PREDICTIONS``: a pickle dictionary where the output values of the model are saved and on which a vote will be applied
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``FOLDS``: a pickle dictionary containing the folds
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``EPOCHS``: number of epochs to train the model
* ``LENGTH``: length of the path to consider for pruning
* ``PADDING``: type of zero padding (before, between, or after the path between seed QID and sub-class)
* ``UNITS``: number of units in LTSM
* ``LEARNING_RATE``: learning rate

#### ``lstm_voting.py``

Applies a vote on the predictions of the LSTM model.

```
python lstm_voting.py --wikidata WIKIDATA --embeddings EMBEDDINGS --folds FOLDS --decisions DECISIONS --predictions PREDICTIONS --voting-threshold THRESHOLD --output OUTPUT
```

with
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``FOLDS``: a pickle dictionary containing the folds
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``PREDICTIONS``: a pickle dictionary where the output values of the model are saved and on which a vote will be applied
* ``THRESHOLD``: value in [0, 1] that takes the decision if an analogy is either valid or invalid
* ``OUTPUT``: path of the pickle output file where results will be stored

#### ``mlp_pruning.py`` and ``mlp_transfer.py``

Performs the same expansion as ``down_expansion.py`` from a set of seed QIDs but prunes sub-classes based on MLP model which is trained on a set of labeled decisions. The MLP model takes as input either the concatenation of the embedding of the seed QID and the embedidng of the sub-class, or the translation between these two embeddings.
The script ``mlp_transfer.py`` is intended for use in a transfer setting.

```
python mlp_pruning.py --wikidata WIKIDATA --embeddings EMBEDDINGS --folds FOLDS --decisions DECISIONS --epochs EPOCHS --concatenation CONCATENATION --hidden-layers LAYERS --learning-rate LEARNING_RATE --dropout DROPOUT --predictions-output PREDICTIONS
```

with
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``FOLDS``: a pickle dictionary containing the folds
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``EPOCHS``: number of epochs to train the model
* ``CONCATENATION``: type of embeddings concatenation (translation or horizontal)
* ``LAYERS``: number of hidden layers and number of neurons of each layer
* ``LEARNING_RATE``: learning rate
* ``DROPOUT``: rate of neurons dropout after each hidden layer
* ``PREDICTIONS``: a pickle dictionary where the output values of the model are saved and on which a vote will be applied

#### ``mlp_voting.py``

Applies a vote on the predictions of the mlp model.

```
python mlp_voting.py --wikidata WIKIDATA --embeddings EMBEDDINGS --folds FOLDS --decisions DECISIONS --predictions PREDICTIONS --voting-threshold THRESHOLD --output OUTPUT
```

with
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``FOLDS``: a pickle dictionary containing the folds
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``PREDICTIONS``: a pickle dictionary where the output values of the model are saved and on which a vote will be applied
* ``THRESHOLD``: value in [0, 1] that takes the decision if an analogy is either valid or invalid
* ``OUTPUT``: path of the pickle output file where results will be stored

#### ``random_forest_pruning.py`` and ``random_forest_transfer.py``

Performs the same expansion as ``down_expansion.py`` from a set of seed QIDs but prunes sub-classes based on MLP model which is trained on a set of labeled decisions. The MLP model takes as input either the concatenation of the embedding of the seed QID and the embedding of the sub-class, or the translation between these two embeddings.
The script ``random_forest_transfer.py`` is intended for use in a transfer setting.

```
python random_forest_pruning.py --wikidata WIKIDATA --seed-qids QIDS --embeddings EMBEDDINGS --decisions DECISIONS --epochs EPOCHS --concatenation CONCATENATION --output-statistics OUTPUT
```

with
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path of the CSV file where statistics of the down expansion will be stored
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``EPOCHS``: number of epochs to train the model
* ``CONCATENATION``: type of embeddings concatenation (translation or horizontal)

#### ``sequence_analogy_pruning.py`` and ``sequence_analogy_transfer.py``

Performs the same expansion as ``down_expansion.py`` from a set of seed QIDs but prunes sub-classes with a sequence analogy-based convolutional model.
The script ``sequence_analogy_transfer.py`` is intended for use in a transfer setting.

```
python sequence_analogy_pruning.py --wikidata WIKIDATA --embeddings EMBEEDINGS --folds FOLDS --decisions DECISIONS --epochs EPOCHS --distances-hashmap DISTANCES --knn KNN --nb-training-analogies-per-decision TRAINING_ANALOGIES --nb-test-analogies TEST_ANALOGIES --nb-keeping-in-test KEEPING_ANALOGIES --nb-pruning-in-test PRUNING_ANALOGIES --sequenced-decisions SEQUENCED_DECISIONS --sequence-length SEQ_LENGTH --padding PADDING --analogical-properties PROPERTIES --valid-analogies-pattern VALID_PATTERN --invalid-analogies-pattern INVALID_PATTERN --dropout DROPOUT --nb-filters1 FILTERS1 --nb-filters2 FILTERS2 --learning-rate LEARNING_RATE --predictions-output PREDICTIONS
```

with
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``FOLDS``: a pickle dictionary containing the folds
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``EPOCHS``: number of epochs to train the model
* ``VOTING``: strategy of voting (majority or weighted)
* ``THRESHOLD``: value in [0, 1] that takes the decision if an analogy is either valid or invalid
* ``DISTANCES``: path to the folder in which the pairwise distances between seed QIDs LMDB hashmap is stored
* ``KNN``: apply K-Nearest-Neighbors on the train set, the test set, both or none
* ``TRAINING_ANALOGIES``: number of training analogies per decision
* ``TEST_ANALOGIERS``: number of testing analogies per decision
* ``KEEPING_ANALOGIES``: number of keeping analogies in test set to evaluate the model on the expansion
* ``PRUNING_ANALOGIES``: number of pruning analogies in test set to evaluate the model on the expansion
* ``SEQ_LENGTH``: length of the sequences to consider
* ``PADDING``: padding mode (before, between, or after)
* ``PROPERTIES``: properties to consider to build the analogies (training and test set) among relfexivity, inner-symmetry, or symmetry
* ``VALID_PATTERN``: valid pattern of analogies
* ``INVALID_PATTERN``: invalid pattern of analogies
* ``DROPOUT``: rate of neurons dropout after each convolutional layer
* ``FILTERS1``: number of filters in first convolutional layer
* ``FILTERS2``: number of filters in second convolutional layer
* ``LEARNING_RATE``: learning rate of the convolutional model
* ``PREDICTIONS``: a pickle dictionary where the output values of the model are saved and on which a vote will be applied

#### ``svm_pruning.py`` and ``svm_transfer.py``

Performs the same expansion as ``down_expansion.py`` from a set of seed QIDs but prunes sub-classes based on MLP model which is trained on a set of labeled decisions. The MLP model takes as input either the concatenation of the embedding of the seed QID and the embedidng of the sub-class, or the translation between these two embeddings.
The script ``svm_transfer.py`` is intended for use in a transfer setting.

```
python svm_pruning.py --wikidata WIKIDATA --seed-qids QIDS --embeddings EMBEDDINGS --decisions DECISIONS --epochs EPOCHS --concatenation CONCATENATION --output-statistics OUTPUT
```

with
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path of the CSV file where statistics of the down expansion will be stored
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``DECISIONS``: set of labeled decisions which is used for training the model
* ``EPOCHS``: number of epochs to train the model
* ``CONCATENATION``: type of embeddings concatenation (translation or horizontal)

#### ``threshold_pruning.py``

Performs a downward expansion and pruning from a set of seed QIDs. 
Specifically, starting from seed QIDs, their direct classes will be retrieved through 1 hop P31 and P279 edges. 
From them, only P279 edges will be followed when expanding until reaching leaves.
Classes will be pruned according to absolute node degree threshold, relative node degree threshold, and 
relative distance in the embedding space (only on labeled decisions).

```
python threshold_pruning.py --nd-threshold ND-THRESHOLD --alpha ALPHA --gamma GAMMA --beta BETA --wikidata WIKIDATA --embeddings EMBEDDINGS --folds FOLDS --output OUTPUT
```

with
* ``ND-THRESHOLD``: absolute degree threshold
* ``ALPHA``: coefficient controlling the relative degree threshold
* ``GAMMA``: minimum degree at an expansion level to enable the relative degree threshold
* ``BETA``: coefficient controlling the relative distance threshold
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``EMBEDDINGS``: path to the folder in which the embedding LMDB hashmap is stored
* ``FOLDS``: a pickle dictionary containing the folds to test on (only the train set will be used)
* ``DECISIONS``: CSV file containing labeled decisions
* ``OUTPUT``: path of the pickle output file where results will be stored

### In ``utils`` folder

#### ``add_depth_to_decisions.py``

Adds a column "depth" to the file containing labeled decisions, corresponding to the depth of the QID reached from the seed QID.

```
python clean_gold_decisions.py --decisions DECISIONS --wikidata WIKIDATA --output OUTPUT
```

with:
* ``WIKIDATA``: path to the Wikidata LMDB hashmap
* ``DECISIONS``: path to the CSV file containing pruning and keeping decisions
* ``OUTPUT``: path to the output CSV file that contains a new column "depth"

#### ``clean_gold_decisions.py``

Cleans labeled decisions by removing those that cannot be reached (because a prune decision is reached before in the expansion).

```
python clean_gold_decisions.py --decisions DECISIONS --wikidata WIKIDATA --output OUTPUT
```

with:
* ``WIKIDATA``: path to the Wikidata LMDB hashmap
* ``DECISIONS``: path to the CSV file containing pruning and keeping decisions
* ``OUTPUT``: path to the CSV file containing cleaned decisions

#### ``clean_gold_decisions_embeddings.py``

Cleans labeled decisions by removing those that cannot be reached (because a prune decision is reached before in the expansion) and those that do not have an embedding.

```
python clean_gold_decisions.py --decisions DECISIONS --wikidata WIKIDATA --embeddings EMBEDDINGS --output OUTPUT
```

with:
* ``WIKIDATA``: path to the Wikidata LMDB hashmap
* ``DECISIONS``: path to the CSV file containing pruning and keeping decisions
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``OUTPUT``: path to the CSV file containing cleaned decisions

#### ``decision_statistics.py``

Compute statistics on decisions (output of the classifers).

```
python decision_statistics.py --classifier-decisions CLASSIFIER_DECISIONS --gold-decisions GOLD_DECISIONS --output OUTPUT
```

with:
* ``CLASSIFIER_DECISIONS``: path of the pickle file where results of the classifier are stored
* ``GOLD_DECISIONS``: path to the CSV dataset containing pruning and keeping decisions
* ``OUTPUT``: Path to the output CSV file containing the average evaluation of the classifier on the folds

#### ``decision_statistics_seen_unseen.py``

Compute statistics on decisions (output of the classifers) by splitting the test set in two sets: an unseen set containing decisions in which classes reached are not seen in the training set and a seen set containing decisions in which classes reached are seen in the training set.

```
python decision_statistics.py --classifier-decisions CLASSIFIER_DECISIONS --wikidata WIKIDATA --gold-decisions GOLD_DECISIONS --folds FOLDS --output OUTPUT
```

with:
* ``CLASSIFIER_DECISIONS``: path of the pickle file where results of the classifier are stored
* ``WIKIDATA``: path to the Wikidata LMDB hashmap
* ``GOLD_DECISIONS``: path to the CSV dataset containing pruning and keeping decisions
* ``FOLDS``: a pickle dictionary containing the folds
* ``OUTPUT``: Path to the output CSV file containing the average evaluation of the classifier on the folds

#### ``generate_sequenced_decisions.py``

Returns a pickle file containing a dictionary associating each seed QID with the traversed QIDs the path containing the embeddings. 
Specifically, the dictionary is structured as follows:
```
dict(
  "seed QID1": dict(
    "traversed QID": array[embedding(QID1), embedding(intermediate QID), ..., embedding(traversed QID) ],
    ...
  ),
  "seed QID2": ...
)
```

```
python generate_sequenced_decisions.py --decisions DECISIONS --wikidata WIKIDATA --embeddings EMBEDDINGS --output OUTPUT
```

with:
* ``DECISIONS``: path to the CSV dataset containing pruning and keeping decisions
* ``WIKIDATA``: path to the Wikidata LMDB hashmap
* ``EMBEDDINGS``: path to the folder in which the embeddings LMDB hashmap is stored
* ``OUTPUT``: Path to the output pickle dictionary

#### ``get_labels.py``

Returns the labels of a list of QIDs.

```
python get_labels.py --qids QIDS --wikidata WIKIDATA --output OUTPUT
```

with:
* ``QIDS``: a CSV file containing one QID per line (not the complete URL, just the QID, no header)
* ``WIKIDATA``: path to the folder in which the Wikidata LMDB hashmap is stored
* ``OUTPUT``: path to the output CSV file containing on each line a QID and its corresponding label

#### ``kfold.py``

Generate k folds from the labeled decisions.

```
python kfold.py --nb-fold K --decisions DECISIONS --output OUTPUT
```

with:
* ``K``: number of folds to generate
* ``DECISIONS``: path to the CSV dataset containing pruning and keeping decisions
* ``OUTPUT``: path to the output pickle file containing the generated folds

#### ``qids_distance_compute.py``

Generate a LMDB hashmap containing the distance in the embedding space between each pair of QIDs in the input set

```
python qids_distance_compute.py --qids QIDS --embeddings EMBEDDINGS --lmdb-size LMDB_SIZE --output OUTPUT
```

with:
* ``QIDS``: CSV file containing QIDs of interest (one QID per line)
* ``EMBEDDINGS``: Folder containing the LMDB embeddings hashmap
* ``LMDB_SIZE``: Size of the output lmdb hashmap
* ``OUTPUT``: Path to the output hashmap

## Citing

When citing, please use the following reference:

```
@inproceedings{jarnacCM2023,
  author       = {Lucas Jarnac and 
                  Miguel Couceiro and 
                  Pierre Monnin},
  title        = {Relevant Entity Selection: Knowledge Graph Bootstrapping via Zero-Shot Analogical Pruning},
  booktitle    = {Proceedings of the 32nd {ACM} International Conference on Information and Knowledge Management (CIKM '23), October 21--25, 2023, Birmingham, United Kingdom},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3583780.3615030},
  doi          = {10.1145/3583780.3615030},
}
```

## Maintainers

* [Lucas Jarnac](mailto:lucas.jarnac@orange.com)
* [Miguel Couceiro](https://members.loria.fr/mcouceiro/)
* [Pierre Monnin](https://pmonnin.github.io/)

## License 

Code and documentation are under the [MIT License](LICENSE.txt) and datasets in the dataset folder are available under the [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license.


## References

1. Lucas Jarnac, Pierre Monnin. Wikidata to Bootstrap an Enterprise Knowledge Graph: How to Stay on Topic?. Proceedings of the 3rd Wikidata Workshop 2022 co-located with the 21st International Semantic Web Conference (ISWC2022), Virtual Event, Hanghzou, China, October 2022.
