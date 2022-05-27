import os
import logging

import hydra
import mlflow

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

@hydra.main(config_name="config", config_path = ".", version_base = None)
def go(config):

    if config["mode"] == "remote":
        os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
        os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()

    steps = config["main"]["steps"]
    to_run = steps.split(",") if steps != "all" else config["components"].keys()

    for component, params in config["components"].items():
        if component in to_run:
            print(f"\n====> Running component: {component}\n")
            mlflow.run(
                os.path.join(root_path, f"components/{component}"),
                "main",
                parameters = {
                    "mode": config["mode"],
                    **params
                }
            )

if __name__ == "__main__":
    go()