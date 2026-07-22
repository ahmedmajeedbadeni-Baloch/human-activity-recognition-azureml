# Human Activity Recognition

A neural network that classifies human physical activity (walking, walking upstairs, walking downstairs, sitting, standing, laying) from smartphone accelerometer and gyroscope readings, with an Azure ML training job, a containerized scoring service, and CI/CD workflows that automate training and container builds.

## Quick start

Clone the repo, then run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook/data_exploration.ipynb
```

Run all cells in the notebook. It loads `data/raw/dataset.csv`, splits it, trains the model, and reproduces the results below.

## What this is

The dataset is the Kaggle "Human Activity Recognition with Smartphones" set, a preprocessed derivative of the UCI HAR Dataset. Each row is one 2.56-second window of accelerometer and gyroscope signal from a smartphone worn at the waist, reduced to 561 time-domain and frequency-domain features (`tBodyAcc-mean()-X`, `fBodyGyro-std()-Z`, and so on), labeled with one of six activities.

`data/raw/dataset.csv` here is the 2947-row test partition of that dataset, not the full 10,299-row set (7352 train + 2947 test). The notebook and `model.py` take that 2947-row file and do their own 80/20 train/test split on it. That means the accuracy below is not a standard UCI HAR benchmark result, since that benchmark trains on the full 7352-row train set and evaluates on the full 2947-row test set. This result is an 80/20 split of a smaller subset, worth knowing before comparing it to published UCI HAR benchmarks.

The model itself is a small feed-forward network: one hidden dense layer of 64 units with sigmoid activation, 20% dropout, and a softmax output layer over the six activity classes. It's trained with the Adam optimizer and sparse categorical crossentropy for 20 epochs, batch size 64.

## Results

From the saved run in `notebook/data_exploration.ipynb`. These numbers come directly from that notebook's saved cell output and were not re-run in this pass.

| Epoch | Train accuracy | Train loss | Val accuracy | Val loss |
|---|---|---|---|---|
| 1 | 0.297 | 1.705 | 0.541 | 1.366 |
| 10 | 0.858 | 0.487 | 0.907 | 0.422 |
| 20 | 0.927 | 0.237 | 0.961 | 0.185 |

Final validation accuracy: 96.1%. This is a single run with a fixed random seed (`random_state=42` for the split), not averaged over multiple runs, and evaluated on the 590-row held-out portion of the 2947-row subset described above.

The Keras Tuner hyperparameter search that was in an earlier version of this repo is not included here. Its best recorded trial scored 0.34, well below what the fixed architecture above achieves, and it never finished running.

## Tech stack

Python 3.12, TensorFlow/Keras 2.16 for the model, scikit-learn for preprocessing (train/test split, label encoding, min-max scaling) and evaluation (classification report, confusion matrix, permutation importance), pandas and numpy for data handling, matplotlib and seaborn for the exploratory and evaluation plots. MLflow and the Azure ML SDK (`azureml-core`) are used in `src/training/model.py` for experiment tracking and to submit the training run as an Azure ML job (`config/job.yaml`). The scoring service is packaged with `azureml-inference-server-http` and Docker for container-based deployment, with GitHub Actions handling both the training job submission and the container build/push, and Kubernetes manifests for running the container on a cluster.

## Repo structure

```
notebook/data_exploration.ipynb     Data loading, preprocessing, model training and evaluation (the source of the results above)
src/training/model.py               Training script version of the notebook, meant to run as an Azure ML job
src/training/score.py               Scoring/inference entry point for deploying the trained model as an Azure ML endpoint
config/job.yaml                     Azure ML job definition that runs model.py on Azure ML compute
config/aml.environment.yml          Azure ML environment definition, points at config/environment.yml
config/environment.yml              Conda/pip environment spec used by the Azure ML job
data/raw/dataset.csv                The 2947-row dataset described above
requirements.txt                    Python dependencies for running the notebook and scripts locally
Dockerfile                          Builds the container that serves score.py as an HTTP inference endpoint
kubernetes/deployment.yaml          Kubernetes Deployment for running the container on a cluster
kubernetes/service.yaml             Kubernetes Service that exposes the deployment
.github/workflows/train-deploy.yml       GitHub Actions workflow that submits config/job.yaml to Azure ML on push to main
.github/workflows/docker-build-push.yml  GitHub Actions workflow that builds the Dockerfile and pushes it to Docker Hub on push to main
```

## Running it

**Locally, just to see the model train (no Azure account needed):** use the Quick start commands above. This is the fastest way to reproduce the results table.

**As an Azure ML job (what this project is actually set up for):** `src/training/model.py` calls `Workspace.from_config()` and logs to MLflow through an Azure ML workspace, so running it directly requires an Azure ML workspace and a local `config.json` for that workspace. That file is not included in this repo; see the [Azure ML docs](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment) for how to generate one. With that in place:

```bash
az extension add -n ml
az ml job create --file config/job.yaml --resource-group Com774-Ahmed --workspace-name Com774-Ahmed
```

This expects a dataset registered in the workspace as `HAR_dataset:1`. Update `config/job.yaml` if your workspace uses a different name.

**Via GitHub Actions:** `.github/workflows/train-deploy.yml` runs the command above automatically on push to `main`. It needs a repository secret named `AZURE_CREDENTIALS` (a service principal with access to the workspace) to authenticate. This workflow is new as of this cleanup pass and has not been run end to end against a live workspace yet.

**As a container:** the `Dockerfile` builds an image that serves `score.py` over HTTP using Azure ML's `azmlinfsrv` tool, on port 5001. It does not bake a trained model into the image; the image expects a model directory to be provided at runtime through `AZUREML_MODEL_DIR`, either by Azure ML's own container deployment (which mounts the registered model automatically) or, if you run it yourself, by mounting one in:

```bash
docker build -t human-activity-recognition .
docker run -p 5001:5001 -v /path/to/your/exported/model:/app/model -e AZUREML_MODEL_DIR=/app/model human-activity-recognition
```

`.github/workflows/docker-build-push.yml` builds this image and pushes it to Docker Hub on push to `main`. It needs two repository secrets, `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (an access token, not your account password, generated from Docker Hub under Account Settings > Security).

**On Kubernetes:** `kubernetes/deployment.yaml` and `kubernetes/service.yaml` run the container on a cluster. Update the image reference in `deployment.yaml` to your actual Docker Hub username, and see the comments in that file about the model volume: a plain Kubernetes cluster doesn't have Azure ML's automatic model-mounting, so you need to provide the trained model to the pod yourself (a PersistentVolumeClaim populated with the exported model, or your own init container that pulls it from wherever you store it). Once that's set up:

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

## What still needs verifying

A few things in this repo were fixed, rebuilt, or added during cleanup and have not been re-run against real infrastructure:

`model.py` and `score.py` had missing function bodies before this pass (the file contained placeholder comments instead of actual code) and have been reconstructed from the notebook's verified logic. The preprocessing half (`preprocess_data`) has been re-run and produces the same shapes as the original notebook, but the full training loop, including the new callbacks and evaluation additions below, has not been re-run end to end since the fix.

`model.py` now also has early stopping and best-checkpoint saving (`EarlyStopping`, `ModelCheckpoint`), a per-class classification report and confusion matrix after training, and a permutation-importance ranking of the 561 input features. None of this has been executed; it needs a real run to confirm it works and to see what the actual per-class and feature-importance results look like.

`score.py` is a freshly written scoring script for an Azure ML endpoint deployment. It has not been deployed or tested against a live endpoint.

`.github/workflows/train-deploy.yml` is new and has not been triggered against a real Azure ML workspace.

`Dockerfile`, `.github/workflows/docker-build-push.yml`, and both `kubernetes/*.yaml` manifests are new in this pass. None of them have been built, pushed, or applied to a real cluster. The Kubernetes manifests in particular assume a model-delivery mechanism (a PersistentVolumeClaim named `har-model-pvc`) that doesn't exist yet and that you'll need to set up or replace before `kubectl apply` will actually work.

If you run any of these and they work as described, that closes the loop. If something breaks, the error message will point at exactly what needs fixing next.
