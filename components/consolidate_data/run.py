from argparse import ArgumentParser
import logging
import os
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile

import pandas as pd

from utils.general import create_artifact_folder

logger = logging.getLogger()

def consolidate_data(args):
    cols = [
        "Date", "Page Type", "Account Type", "Author", "Full Text", 
        "Gender", "Hashtags", "Impact", "Impressions", "Thread Entry Type", 
        "Twitter Followers", "Twitter Following", "Twitter Tweets", "Twitter Verified", 
        "Reach (new)", "Region"
    ]

    artifact_path = create_artifact_folder(__file__)

    logger.info("Combining data files into single CSV...")

    try:
        with ZipFile(args.input_path, "r") as zip: 
            with open(artifact_path / "metoo_data.csv", "w") as outfile:
                outfile.write("\t".join(cols) + "\n")
                
            r = re.compile("^All Raw Data\/.{1,}\.xlsx$")

            for file in zip.namelist():
                if r.match(file):
                    data = pd.read_excel(zip.read(file), header = 6, usecols = cols, engine = "openpyxl")
                    data = data[data["Page Type"] == "twitter"]
                    data.to_csv(
                        path_or_buf = artifact_path / "metoo_data.csv", 
                        sep = "\t", 
                        header = False, 
                        mode = "a", 
                        index = False
                    )

    except Exception as e:
        logger.error("An error occurred in the data consolidation process.")
        raise e

    if args.mode == "remote":
        import wandb
        from wandb_utils.utils import log_artifact

        run = wandb.init(job_type = "consolidate_data")
        logger.info("Logging artifact in W&B...")
        log_artifact(
            run = run, 
            path_to_file = artifact_path / "metoo_data.csv",
            art_name = "metoo_data.csv",
            art_type = "csv",
            art_desc = "Consolidated MeToo twitter data"
        )

def go(args):
    if os.path.exists(args.input_path):
        consolidate_data(args)
    else:
        logger.error("Downloaded data not found in target location.")
        raise FileNotFoundError


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to input data (.zip)")      
    args = parser.parse_args()

    go(args)