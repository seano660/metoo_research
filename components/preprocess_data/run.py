from argparse import ArgumentParser
import importlib
import logging
import re

import nltk
nltk.download("stopwords", quiet = True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

from component_utils.general import create_artifact_folder

spark = SparkSession.builder.getOrCreate()

logger = logging.getLogger()


def go(args):
    artifact_path = create_artifact_folder(__file__)

    mention_udf = udf(lambda x: re.sub("@\S*\s?", "", x), StringType())
    link_udf = udf(lambda x: re.sub("http\S*\s?", "", x), StringType())
    

    logger.info("Reading data from input...")
    data = (
        spark.read.options(sep = "\t", header = True, usecols = ["Full Text", "Thread Entry Type"])
        .csv(args.input_path)
        .repartition(10)
        .filter("`Thread Entry Type` != 'share'")
        .select("`Full Text`")
        # .withColumn("full_text", mention_udf(col("`Full Text`")))
        # .withColumn("full_text", link_udf(col("full_text")))
        .write.csv(str(artifact_path / "processed_data.csv"))
    )

    # text_raw = data[data["Thread Entry Type"] != "share"]["Full Text"].astype(str) # remove retweets

    # data = pd.read_csv(args.input_path, sep = "\t")
    # text_raw = data[data["Thread Entry Type"] != "share"]["Full Text"] # remove retweets

    # sws = stopwords.words("english")
    # exs = args.exclude.split(",")
    # if len(exs) > 0:
    #     sws.extend(exs)

    # sws = set(sws)

    # text_processed = (
    #     text_raw.apply(lambda x: re.sub("@\S*\s?", "", x)) # remove mentions
    #     .apply(lambda x: re.sub("http\S*\s?", "", x)) # remove links
    #     .apply(lambda x: re.sub("\s+", "", x)) # remove newlines
    #     .apply(lambda x: re.sub("\'", "", x)) # remove single quotes
    #     .apply(lambda x: " ".join([w for w in word_tokenize(x) if w not in sws])) # remove stopwords
    # )

    # text_processed.to_spark().to_csv(artifact_path / "processed_data.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)") 
    parser.add_argument("--exclude", type = str, default = "", help = "Additional (comma-separated) stopwords to exclude")       
    args = parser.parse_args()

    go(args)