from argparse import ArgumentParser
import logging
import re

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
    data = pd.read_csv(args.input_path, sep = "\t", usecols = ["Full Text", "Thread Entry Type"])

    data["Full Text"] = (
        data[data["Thread Entry Type"] != "share"]
        ["Full Text"].astype(str)
        .applymap(lambda x: re.sub("http\S*\s?", "", x)) # remove links
        .applymap(lambda x: re.sub("\s+", "", x)) # remove newlines
        .applymap(lambda x: re.sub("\'", "", x)) # remove single quotes
        .applymap(lambda x: " ".join([w for w in simple_preprocess(x, deacc = True) if w not in sws])) # remove stopwords
    )

    data.to_csv(artifact_path / "metoo_data.csv", sep = "\t")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)") 
    parser.add_argument("exclude", type = str, default = "", help = "Additional (comma-separated) stopwords to exclude")       
    args = parser.parse_args()

    go(args)