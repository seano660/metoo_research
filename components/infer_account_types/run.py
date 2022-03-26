from argparse import ArgumentParser
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

from utils.general import create_artifact_folder

logger = logging.getLogger()

def go(args):
    artifact_path = create_artifact_folder(__file__)

    logger.info("Reading input file...")
    data = pd.read_csv(args.input_path, sep = "\t")

    logger.info("Training NB model...")
    author_data = (
        data.groupby(["Author", "Account Type"])["full_text"]
        .transform(" ".join)
        .reset_index()
    )

    vec = TfidfVectorizer(max_features = args.vocab_size)

    adata_l = vec.fit_transform(author_data[author_data["Account Type"].notna()])
    adata_ul = vec.transform(author_data[author_data["Account Type"].isna()])

    X_train, X_test, y_train, y_test = train_test_split(
        adata_l["full_text"], 
        adata_l["Account Type"],
        stratify = adata_l["Account Type"],
        random_state = args.random_state
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    cnf = confusion_matrix(y_test, model.predict(X_test))
    print(cnf)

    ul_preds = model.predict(adata_ul)
    mapper = {user: label for user, label in zip(adata_ul["Author"], ul_preds)}

    data.loc[data["Account Type"].isna(), "Account Type"] = data.loc[data["Account Type"].isna(), "Account Type"].map(mapper)

    data.to_csv(artifact_path / "metoo_data.csv", sep = "\t")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)")
    parser.add_argument("--random_state", type = int, default = None, help = "Seed for setting random state")
           
    args = parser.parse_args()

    go(args)