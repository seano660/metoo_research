from argparse import ArgumentParser
import logging
import re

import nltk
nltk.download(["stopwords", "punkt"], quiet = True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

from component_utils.general import create_artifact_folder

logger = logging.getLogger()


def go(args):
    artifact_path = create_artifact_folder(__file__)

    sws = stopwords.words("english")
    exs = args.exclude.split(",")
    if len(exs) > 0:
        sws.extend(exs)

    sws = set(sws)
    
    logger.info("Reading data from input file...")
    data = pd.read_csv(args.input_path, sep = "\t")
        
    data = (
        [data["Thread Entry Type"] != "share"]["Full Text"].astype(str)
        .applymap(lambda x: re.sub("http\S*\s?", "", x)) # remove links
        .applymap(lambda x: re.sub("\s+", "", x)) # remove newlines
        .applymap(lambda x: re.sub("\'", "", x)) # remove single quotes
        .applymap(lambda x: " ".join([w for w in word_tokenize(x) if w not in sws])) # remove stopwords
        .to_csv(artifact_path / "metoo_data.csv", sep = "\t")
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)") 
    parser.add_argument("--exclude", type = str, default = "", help = "Additional (comma-separated) stopwords to exclude")       
    args = parser.parse_args()

    go(args)