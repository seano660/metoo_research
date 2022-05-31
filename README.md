# metoo_research
(Northeastern) Research on social media patterns surrounding MeToo movement

---

### Setup

The code builds its own isolated Conda environments to run each component of the pipeline, and thus requires minimal manual setup. 

**Note**: The `infer_demographics` component requires an older version of tensorflow which is not compatible with Apple M1 chips. Affected users will not be able to run this component. 

#### conda
`conda` is a basic requirement of any system looking to run the code: see [conda's install docs](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for instructions on how to install for your system, and [Northeastern's RC-DOCS](https://rc-docs.northeastern.edu/en/latest/software/conda.html#working-with-a-miniconda-environment) for info more specifically tailored to the Discovery Cluster. 

It is recommended to create a new `conda` environment to run the code (see [conda's documentation](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments) and [Northeastern's RC-DOCS](https://rc-docs.northeastern.edu/en/latest/software/conda.html#working-with-a-miniconda-environment)).

**NOTE**: If using the Discovery Cluster, `miniconda` is recommended over `conda` for minimal friction with current batch scripts.

#### Requirements
Once in the desired environment, any requisite installs can be installed through the following command (ensure you've navigated to the root directory of the code):

```
conda install --file requirements.txt --channel conda-forge
```

#### Data
The `data` folder hosts two files used for tagging business entities, but this folder should also be the location for the raw data ZIP file (NOT available from this repository). 

---

### Run the code
The project is built using `mlflow` and `hydra`, with a central `config.yaml` file that sets key pipeline parameters. Most components will take some kind of input, which is generally the output of the preceding component; other components have parameters that can be set/modified here as well. 

To change the value of any parameter at runtime, use the optional `-P` flag followed by the desired key/value. 

**NOTE**: Runtimes will be much longer when running the code from scratch, as it takes some time to configure/install the conda environments of each component. 

#### In interactive terminal (local environment)

From the root directory of the code, you can run the entire pipeline using the following command: 
```
mlflow run .
```

To run just one component (or a subset of them):

```
mlflow run . -P steps=component1,component2,...
```

#### In Discovery Cluster

It is not recommended to run Discovery Cluster jobs in the same way as the local environment, because any connection interruptions will automatically terminate the session. Batch scripts should be used instead, which run independently of the local session. There are several pre-configured batch scripts in the `scripts` directory: `main.script` will run the entire pipeline, and other scripts run a single component. These scripts should be run using the `sbatch` command (see [RC-DOCS](https://rc-docs.northeastern.edu/en/latest/using-discovery/sbatch.html)); for example:
```
sbatch scripts/main.script
```

For debugging purposes, job results will be output in the `runs` directory of the root folder. 

---

### Notebooks
Analysis notebooks are also included in the `notebooks` directory. 
* `lda_vis` is meant as a testing ground for selecting the best LDA model
* `analysis` contains any analysis on the final pipeline artifact (i.e. processed/labeled tweets with topic scores). 

If running these notebooks in the Discovery Cluster, it is recommended to specify the conda PATH when starting the notebook compute cluster.
