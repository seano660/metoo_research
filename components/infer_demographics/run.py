from argparse import ArgumentParser
import re
from typing import List

from demographer import process_tweet
import pandas as pd

from component_utils.general import create_artifact_folder


def get_demographics(user_data: pd.Series):
    preds = process_tweet({
        "name": user_data["name"].strip(),
        "username": user_data["Author"].strip(),
        "follower_count": user_data["Twitter Followers"],
        "friends_count": user_data["Twitter Following"],
        "verified": user_data["Twitter Verified"],
        "statuses_count": user_data["Twitter Tweets"]
    })

    return [
        preds["gender_neural"]["value"],
        preds["indorg_neural"]["value"]
    ]


def go(args):
    artifact_path = create_artifact_folder(__file__)

    data = pd.read_csv(args.input_path, sep = "\t")
    data["name"] = (
        data["Full Name"].str
        .extract("\((.{1,})\)") # extract text between parens
        .replace("#[A-z]{1,}\s+", "") # remove hashtags
        .replace("[^A-z]", "") # remove non-alphanumeric
    )

    remap = {"man": "male", "woman": "female"}

    data[["gender_inf", "indorg_inf"]] = data.apply(get_demographics, axis = 1)
    data["gender_inf"] = data["gender_inf"].map(remap)

    data.to_csv(artifact_path / "metoo_data.csv", sep = "\t")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)")      
    args = parser.parse_args()

    go(args)