from argparse import ArgumentParser
import logging

import pandas as pd
from gensim.utils import simple_preprocess
import spacy

from component_utils.general import create_artifact_folder

logger = logging.getLogger()


def go(args):
    artifact_path = create_artifact_folder(__file__)

    nlp = spacy.load("en_core_web_sm")
    sws = nlp.Defaults.stop_words
    exs = args.exclude.split(",")
    if len(exs) > 0:
        sws.update(exs)
    
    logger.info("Reading data from input file...")
    data = pd.read_csv(args.input_path, sep = "\t")
    data = data[data["Thread Entry Type"] != "share"]

    data["Full Text"] = (
        data["Full Text"].fillna("")
        .str
        .replace("http\S*\s?", "") # remove links
        .replace("\s+", " ") # replace any escape character with space
        .replace("'", "") # remove single quotes
        .apply(lambda x: " ".join([w for w in simple_preprocess(x, deacc = True) if w not in sws])) # remove stopwords
    )

    data.to_csv(artifact_path / "metoo_data.csv", sep = "\t")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)") 
    parser.add_argument("exclude", type = str, default = "", help = "Additional (comma-separated) stopwords to exclude")       
    args = parser.parse_args()

    go(args)