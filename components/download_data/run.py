# The ID of the raw data file
DATA_FILE_ID = "1d2ZOvueVgN3x9Cu4XGYcUV-fKBFBq4vB"

from argparse import ArgumentParser
import os
from pathlib import Path
import logging
import requests

from utils.general import create_artifact_folder

logger = logging.getLogger()

def go(args):
    session = requests.Session()

    artifact_path = create_artifact_folder(__file__)

    try:
        response = session.get(
            url = f"https://drive.google.com/uc?id={DATA_FILE_ID}",
            stream = True
        )

        with open(artifact_path / "metoo_data.zip", "wb") as file:
            for chunk in response.iter_content(chunk_size = 32768):
                if chunk:
                    file.write(chunk)

            logger.info("Raw data successfully written to file.")

    except Exception as e:
        logger.info("An exception occurred while reading in the raw data.")
        raise e

    if args.mode == "remote":
        import wandb
        from utils.wandb import log_artifact

        run = wandb.init(job_type = "download_data")

        log_artifact(
            run = run, 
            path_to_file = args.output_path,
            art_name = "metoo_data.zip",
            art_type = "zip",
            art_desc = "Raw MeToo twitter data"
        )

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")        
    args = parser.parse_args()
    go(args)