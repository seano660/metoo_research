import wandb

def log_artifact(run, path_to_file, art_name: str, art_type: str, art_desc: str):
    artifact = wandb.Artifact(
        name = art_name,
        type = art_type,
        description = art_desc
    )
    
    artifact.add_file(path_to_file)
    run.log_artifact(artifact)
    artifact.wait()