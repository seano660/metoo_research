from argparse import ArgumentParser

from demographer import process_tweet
import pandas as pd
import numpy as np

from component_utils.general import create_artifact_folder


def get_demographics(user_data: pd.Series):
    preds = process_tweet({
        "name": user_data["name"].strip(),
        "username": user_data.name.strip(),
        "follower_count": user_data["Twitter Followers"],
        "friends_count": user_data["Twitter Following"],
        "verified": user_data["Twitter Verified"],
        "statuses_count": user_data["Twitter Tweets"]
    })

    return [
        user_data["Gender"] if np.notna(user_data["Gender"]) else preds["gender_neural"]["value"],
        user_data["Account Type"] if np.notna(user_data["Account Type"]) else preds["indorg_neural_full"]["value"]
    ]


def go(args):
    artifact_path = create_artifact_folder(__file__)

    data = pd.read_csv(args.input_path, sep = "\t")

    data["name"] = (
        data["Full Name"].str
        .extract("\((.{1,})\)") # extract text between parens
        .replace("#[A-z]{1,}\s?", "") # remove hashtags
        .replace("[^A-z]", "") # remove non-alphanumeric
    )

    gender_map = {"man": "male", "woman": "female"}
    indorg_map = {"ind": "individual", "org": "organisational"}

    authors = data.groupby("Author").first()

    demo_inf = pd.DataFrame(
        authors.apply(get_demographics, axis = 1),
        columns = ["gender_inf", "indorg_inf"],
        index = authors.index
    )

    demo_inf["gender_inf"] = demo_inf["gender_inf"].replace(gender_map)
    demo_inf["indorg_inf"] = demo_inf["indorg_inf"].replace(indorg_map)

    demo_inf.to_csv(artifact_path / "inferred_demographics.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)")      
    args = parser.parse_args()

    go(args)