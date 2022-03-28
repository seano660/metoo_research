from argparse import ArgumentParser
from typing import List

from demographer.gender_neural import NeuralGenderDemographer
import pandas as pd

from component_utils.general import create_artifact_folder


def _get_gender_mapper(users: List):
    remap = {"man": "male", "woman": "female"}
    ngd = NeuralGenderDemographer()

    return {
        u: remap[
            ngd.process_tweet({"name": u.strip()})["gender_neural"]["value"]
        ]
        for u in users
    }


def go(args):
    artifact_path = create_artifact_folder(__file__)

    data = pd.read_csv(args.input_path, sep = "\t")
    users = [u for u in data["Author"].unique()]

    gender_mapper = _get_gender_mapper(users)

    data["gender_inferred"] = data["Author"].str.strip().map(gender_mapper)

    data.to_csv(artifact_path / "metoo_data.csv", sep = "\t")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)")      
    args = parser.parse_args()

    go(args)