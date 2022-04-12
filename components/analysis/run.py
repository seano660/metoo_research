from argparse import ArgumentParser
from itertools import product
import logging

from gensim.utils import simple_preprocess
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from component_utils.general import create_artifact_folder

logger = logging.getLogger()

def go(args):
    artifact_path = create_artifact_folder(__file__)

    data = pd.read_csv(args.data_path, sep = "\t")
    model = LdaMulticore.load(args.model_path)

    tokenized_data = [simple_preprocess(text) for text in data["Full Text"]]

    dictionary = corpora.Dictionary(tokenized_data)
    dictionary.filter_extremes(no_below = args.no_below, keep_n = args.vocab_size)

    logger.info("Creating corpus...")
    corpus = [dictionary.doc2bow(doc, allow_update=False) for doc in tokenized_data]


        
   

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("data_path", type = str, help = "Path to data")
    parser.add_argument("model_path", type = str, help = "Path to LDA model")
    args = parser.parse_args()

    go(args)