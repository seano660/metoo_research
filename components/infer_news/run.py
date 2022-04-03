from argparse import ArgumentParser

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

from component_utils.general import create_artifact_folder


def go(args):
    artifact_path = create_artifact_folder(__file__)

    data = pd.read_csv(args.input_path, sep = "\t")

    author_corpus = data.groupby("Author")[["Full Text"]].apply(list)
    
    print(author_corpus.head())
    
    author_corpus["news_inf"] = author_corpus.index.str.contains("news").astype(int)

    vec = TfidfVectorizer(max_features = args.vocab_size)
    
    X_train, X_test, y_train, y_test = train_test_split(
        author_corpus["Full Text"], 
        author_corpus["news_inf"],
        random_state = args.random_state,
        train_size = args.train_size
    )

    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    train_preds = pd.Series(model.predict(X_train_vec), index = X_train.index)
    test_preds = pd.Series(model.predict(X_test_vec), index = X_test.index)
    preds = pd.concat([train_preds, test_preds], axis = 1)

    author_corpus["news_inf"].replace({0: np.nan}, inplace = True)
    author_corpus["news_inf"] = author_corpus["news_inf"].combine_first(preds)
    
    cnf = confusion_matrix(y_test, test_preds)
    print(cnf)

    author_corpus["news_inf"].to_csv(artifact_path / "news.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)")
    parser.add_argument("vocab_size", type = int, help = "Max. # of words to include in corpus")
    parser.add_argument("train_size", type = float, help = "Portion of input data to use in training")
    parser.add_argument("random_state", type = int, help = "Seed for setting random state")
    args = parser.parse_args()

    go(args)