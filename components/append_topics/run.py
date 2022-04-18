from argparse import ArgumentParser
import logging
import pickle

import gensim
from gensim.utils import simple_preprocess
from gensim.models.ldamulticore import LdaMulticore
import pandas as pd

from component_utils.general import create_artifact_folder

logger = logging.getLogger()

def go(args):
    artifact_path = create_artifact_folder(__file__)

    data = pd.read_csv(args.data_path, sep = "\t")
    data["Full Text"] = data["Full Text"].fillna("")
    model = LdaMulticore.load(args.model_path)

    tokenized_data = [simple_preprocess(text) for text in data["Full Text"]]
    with open(args.dict_path, "rb") as f:
        dictionary = pickle.load(f)
    corpus = [dictionary.doc2bow(doc, allow_update=False) for doc in tokenized_data]

    topics = [model.get_document_topics(tweet) for tweet in corpus]

    topics_mat = gensim.matutils.corpus2csc(topics).T

    tdf = pd.DataFrame.sparse.from_spmatrix(topics_mat)

    data[[f"topic_{i+1}" for i in range(model.num_topics)]] = tdf

    data.to_csv(artifact_path / "metoo_topics.csv", sep = "\t")        
   

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("data_path", type = str, help = "Path to data")
    parser.add_argument("model_path", type = str, help = "Path to LDA model")
    parser.add_argument("dict_path", type = str, help = "Path to id2word dict")
    args = parser.parse_args()

    go(args)