import os
from pathlib import Path
import shutil

def create_artifact_folder(file):
    artifact_path = Path(file).parent / "artifacts"
    if os.path.exists(str(artifact_path)):
        shutil.rmtree(str(artifact_path))

    os.makedirs(artifact_path)

    return artifact_path