from argparse import ArgumentParser

import pyspark as spark
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from utils.general import create_artifact_folder

def go(args):
    artifact_path = create_artifact_folder(__file__)

    data = spark.read.options(sep = "/t", header = True).csv(args.input_path)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)")      
    args = parser.parse_args()

    go(args)