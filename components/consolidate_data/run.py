from argparse import ArgumentParser
import logging
import re
from zipfile import ZipFile

import pandas as pd

from component_utils.general import create_artifact_folder

logger = logging.getLogger()


def go(args):
    cols = [
        "Date", "Page Type", "Account Type", "Author", "Full Name", "Full Text", 
        "Gender", "Hashtags", "Impact", "Impressions", "Thread Entry Type", "Thread Author",
        "Twitter Followers", "Twitter Following", "Twitter Tweets", "Twitter Reply Count",
        "Twitter Verified", "Twitter Retweets", "Reach (new)", "Region"
    ]

    artifact_path = create_artifact_folder(__file__)

    logger.info("Combining data files into single CSV...")


    with ZipFile(args.input_path, "r") as zip: 
        with open(artifact_path / "metoo_data.csv", "w") as outfile:
            outfile.write("\t".join(cols) + "\n")
            
        r = re.compile("^All Raw Data\/.{1,}\.xlsx$")

        to_compile = [f for f in zip.namelist() if r.match(f)]

        if args.samp_size != -1:
            to_compile = to_compile[:args.samp_size]

        for file in to_compile:
            data = pd.read_excel(zip.read(file), header = 6, usecols = cols, engine = "openpyxl")
            data = data[data["Page Type"] == "twitter"][cols] # Make sure we maintain the correct ordering!
            data.to_csv(
                path_or_buf = artifact_path / "metoo_data.csv", 
                sep = "\t", 
                header = False, 
                mode = "a", 
                index = False
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)") 
    parser.add_argument("--samp_size", type = int, required = False, default = None, help = "Number of files to sample data from")     
    args = parser.parse_args()
    go(args)