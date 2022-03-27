from argparse import ArgumentParser
import logging

from gensim import corpora
from gensim.utils import simple_preprocess
import pandas as pd
from sklearn.model_validation import train_test_split

from component_utils.general import create_artifact_folder

logger = logging.getLogger()

def go(args):
    artifact_path = create_artifact_folder(__file__)

    logger.info("Reading data...")
    data = pd.read_csv(args.input_path)

    logger.info("Tokenizing data...")
    tokenized_data = [simple_preprocess(text) for text in data]

    dictionary = corpora.Dictionary(tokenized_data)
    dictionary.filter_extremes(no_below = args.no_below, keep_n = args.vocab_size)

    corpus = [dictionary.doc2bow(doc, allow_update=False) for doc in tokenized_data]

    logger.info("Splitting train/test data...")
    X_train, X_test = train_test_split(
        corpus, 
        random_state = args.random_state, 
        train_size = args.train_size
    )

    logger.info("Writing train/test data to file...")
    X_train.to_csv(artifact_path / "X_train.csv", sep = "\t")
    X_test.to_csv(artifact_path / "X_test.csv", sep = "\t")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data")     
    parser.add_argument("random_state", type = int, help = "Seed for setting the random state")
    parser.add_argument("vocab_size", type = int, help = "Max. number of words to include in corpus") 
    parser.add_argument("no_below", type = int, help = "Minimum number of occurrences to include in corpus")
    args = parser.parse_args()

    go(args)