from argparse import ArgumentParser

from demographer import process_tweet
import pandas as pd
import numpy as np

from component_utils.general import create_artifact_folder


def get_demographics(user_data: pd.Series):
    preds = process_tweet({"user": user_data.dropna().to_dict()})

    return [preds["gender_neural"]["value"], preds["indorg_neural_full"]["value"]]


def go(args):
    artifact_path = create_artifact_folder(__file__)

    data = pd.read_csv(args.input_path, sep = "\t")

    data["name"] = (
        data["Full Name"]
        .str.extract("\((.{1,})\)", expand = False) # extract text between parens
        .str.replace("[^A-z]", "", regex = True) # remove non-alphanumeric
        .str.strip()
    )

    data["Gender"] = data["Gender"].replace("unknown", None)

    gender_map = {"man": "male", "woman": "female"}
    indorg_map = {"ind": "individual", "org": "organisational"}

    authors = (
        data.sort_values(by = ["Author", "Date"], ascending = [True, False])
        .groupby("Author").first()
        .rename(columns = {
            "Twitter Followers": "followers_count", 
            "Twitter Following": "friends_count",
            "Twitter Verified": "verified",
            "Twitter Tweets": "statuses_count"
            }
        )
    )

    authors.index.name = "screen_name"

    authors[["gender_inf", "indorg_inf"]] = (
        authors[["name", "followers_count", "friends_count", "statuses_count", "verified"]]
        .apply(get_demographics, axis = 1)
    )

    authors["Gender"] = authors["Gender"].combine_first(authors["gender_inf"]).map(gender_map)
    authors["Account Type"] = authors["Account Type"].combine_first(authors["indorg_inf"]).map(indorg_map)

    authors[["Gender", "Account Type"]].to_csv(artifact_path / "inferred_demographics.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)")      
    args = parser.parse_args()

    go(args)