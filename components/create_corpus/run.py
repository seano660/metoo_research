from argparse import ArgumentParser
import logging
from typing import Optional

from gensim import corpora
from gensim.utils import simple_preprocess
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from component_utils.general import create_artifact_folder

logger = logging.getLogger()

def go(args):
    artifact_path = create_artifact_folder(__file__)

    logger.info("Reading data...")
    data = pd.read_csv(args.input_path, sep = "\t", usecols = ["Full Text"])

    logger.info("Tokenizing data...")
    tokenized_data = [simple_preprocess(text) for text in data]

    dictionary = corpora.Dictionary(tokenized_data)
    dictionary.filter_extremes(no_below = args.no_below, keep_n = args.vocab_size)

    logger.info("Creating corpus...")
    corpus = [dictionary.doc2bow(doc, allow_update=False) for doc in tokenized_data]

    logger.info("Splitting train/test corpus...")
    X_train, X_test = train_test_split(
        corpus, 
        random_state = args.random_state, 
        train_size = args.train_size
    )

    logger.info("Writing train/test corpus to file...")
    X_train.to_csv(artifact_path / "X_train.csv", sep = "\t")
    X_test.to_csv(artifact_path / "X_test.csv", sep = "\t")
    pickle.dump(dictionary, artifact_path / "dictionary.obj")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data")
    parser.add_argument("train_size", type = float, help = "Portion of data to set aside for training")
    parser.add_argument("random_state", type = int, default = None, help = "Seed for setting the random state")
    parser.add_argument("vocab_size", type = int, default = None, help = "Max. # of words to include in corpus") 
    parser.add_argument("no_below", type = int, default = None, help = "Min. # of occurrences to include in corpus")
    args = parser.parse_args()

    go(args)