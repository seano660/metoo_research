from setuptools import setup

setup(
    name = "component_utils",
    description = "internal utility functions",
    zip_safe = False,
    packages = ["component_utils"],
    install_requires = [
        "mlflow",
        "wandb"
    ]
)