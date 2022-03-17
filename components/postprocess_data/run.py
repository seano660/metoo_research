from argparse import ArgumentParser
from typing import List

from demographer import NeuralGenderDemographer
import pyspark as spark

from component_utils.general import create_artifact_folder


def _get_gender_mapper(users: List):
    remap = {"Man": "Male", "Woman": "Female"}
    ngd = NeuralGenderDemographer()

    return {
        u: remap[
            ngd.process_tweet({"name": u})["gender_neural"]["value"]
        ]
        for u in users
    }


def go(args):
    artifact_path = create_artifact_folder(__file__)

    data = spark.read.options(sep = "/t", header = True).csv(args.input_path).to_pandas()
    users = [u.strip() for u in data["Author"].unique()]

    gender_mapper = _get_gender_mapper(users)

    data["gender_inferred"] = data["author"].map(gender_mapper)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)")      
    args = parser.parse_args()

    go(args)