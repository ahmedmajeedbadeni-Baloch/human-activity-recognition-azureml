# Custom inference container for deploying the trained HAR model as an
# Azure ML online endpoint (bring-your-own-container deployment).
#
# This image does NOT bake the trained model into it. Azure ML mounts the
# registered model into the running container at deploy time and sets
# AZUREML_MODEL_DIR to point at it, which is what score.py's init() reads
# from. That means this image is safe to build and push before a model has
# even been trained, and the same image can be reused across model versions.
#
# This has not been built, pushed, or deployed in this pass. Build and test
# it locally before wiring it into a real Azure ML endpoint deployment.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# azureml-inference-server-http provides the `azmlinfsrv` command, which
# loads init()/run() from an entry script (score.py here) and serves them
# over HTTP using Azure ML's expected request/response and health-check
# contract. It's kept out of requirements.txt on purpose, since that file is
# also used for the plain notebook/local workflow, which doesn't need it.
RUN pip install --no-cache-dir -r requirements.txt azureml-inference-server-http

COPY src/training/score.py /app/score.py

ENV AZUREML_ENTRY_SCRIPT=score.py

EXPOSE 5001

CMD ["azmlinfsrv", "--entry_script", "score.py", "--port", "5001"]
