import os
from pathlib import Path

def create_artifact_folder(file):
    artifact_path = Path(file).parent / "artifacts"
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path)

    return artifact_path