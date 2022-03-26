from argparse import ArgumentParser
from itertools import product
import logging

import nltk
nltk.download("punkt", quiet = True)

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from gensim.models.ldamodel import LdaMulticore
from gensim import corpora
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from component_utils.general import create_artifact_folder

logger = logging.getLogger()

gridsearch_params = {
    "num_topics": range(25, 201, 25),
    "decay": range(0.5, 1, 0.1)
}

def go(args):
    artifact_path = create_artifact_folder(__file__)

    logger.info("Reading data...")
    data = pd.read_csv(args.input_path)

    logger.info("Tokenizing data...")
    tokenized_data = [word_tokenize(text) for text in data]

    dictionary = corpora.Dictionary(tokenized_data)
    dictionary.filter_extremes(no_below = 10, keep_n = 4000)

    corpus = [dictionary.doc2bow(doc, allow_update=False) for doc in tokenized_data]

    X_train, X_test = train_test_split(corpus, random_state = args.random_state)

    best_model, best_params, best_perp = None, None, np.inf

    for grid_params in dict(zip(gridsearch_params.keys(), product(*gridsearch_params.values()))):
        logger.info(f"Training model with params {grid_params}...")
        model = LdaMulticore(
            corpus = X_train, 
            id2word = dictionary, 
            random_state = args.random_state,
            **grid_params
        )
        
        model_perp = model.log_perplexity(X_test)
        if model_perp < best_perp:
            best_perp = model_perp
            best_params = grid_params
            best_model = model
    
    logger.info(f"Saving best model ({best_params}) to file...")
    best_model.save(artifact_path / "lda_model.model")
        
   

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)")   
    parser.add_argument("--random_state", type = int, default = None, help = "Seed for setting random state")     
    args = parser.parse_args()

    go(args)