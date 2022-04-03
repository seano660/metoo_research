from argparse import ArgumentParser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

from component_utils.general import create_artifact_folder


def go(args):
    artifact_path = create_artifact_folder(__file__)

    X_train = pd.read_csv(args.train_path, sep = "\t")
    X_test = pd.read_csv(args.test_path, sep = "\t")

    vec = TfidfVectorizer()

    X_train_vec = vec.fit_transform(X_train.groupby("Author")["Full Text"].apply(list))
    X_test_vec = vec.transform(X_test.groupby("Author")["Full Text"].apply(list))

    y_train = X_train["Author"].str.contains("news").astype(int)
    y_test = X_test["Author"].str.contains("news").astype(int)

    y = pd.concat([y_train, y_test], axis = 1)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    train_preds = pd.Series(model.predict(X_train_vec), index = X_train.index)
    test_preds = pd.Series(model.predict(X_test_vec), index = X_test.index)
    preds = pd.concat([train_preds, test_preds], axis = 1)

    news_inf = y.replace({0: np.nan}).combine_first(preds)
    
    cnf = confusion_matrix(y_test, test_preds)
    print(cnf)

    news_inf.to_csv(artifact_path / "news_inf.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)")
    args = parser.parse_args()

    go(args)