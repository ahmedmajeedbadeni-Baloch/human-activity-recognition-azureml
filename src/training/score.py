"""
Scoring/inference entry point for deploying the trained HAR model as an
Azure ML online (or ACI/AKS) endpoint.

This follows the standard Azure ML deployment contract: init() runs once
when the endpoint starts and loads the model, run() is called once per
request with the incoming payload and returns predictions.

NOTE: this file has not been deployed/tested against a live Azure ML
endpoint in this pass. It is written to match the model produced by
model.py exactly (same input feature count, same 6 activity classes), but
you should verify it end to end with a real deployment before relying on it.
"""

import os
import json
import numpy as np
from tensorflow.keras.models import load_model

# The six UCI HAR activity labels, in the same alphabetical order that
# sklearn's LabelEncoder assigns during training (see preprocess_data in
# model.py). If you retrain with a different label set, update this list
# to match.
ACTIVITY_LABELS = [
    "LAYING",
    "SITTING",
    "STANDING",
    "WALKING",
    "WALKING_DOWNSTAIRS",
    "WALKING_UPSTAIRS",
]

model = None


def init():
    """
    Called once when the endpoint container starts. Loads the model that
    was saved by model.py's main() (model.save(os.path.join(output_dir, "model"))).

    Azure ML sets AZUREML_MODEL_DIR to the directory containing the
    registered model when this script runs inside a deployment. Locally,
    set that environment variable yourself (or pass a path) to test init().
    """
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "model")
    model = load_model(model_path)


def run(raw_data):
    """
    Called once per request. Expects a JSON body of the form:
        {"data": [[<561 feature values>], [<561 feature values>], ...]}
    where each inner list is one row of the same 561 scaled sensor features
    used during training (apply the same MinMaxScaler used in preprocess_data
    before sending data here).

    Returns a JSON-serializable list of predicted activity labels.
    """
    try:
        payload = json.loads(raw_data)
        input_data = np.array(payload["data"])

        predictions = model.predict(input_data)
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_labels = [ACTIVITY_LABELS[i] for i in predicted_indices]

        return {"predictions": predicted_labels}
    except Exception as e:
        return {"error": str(e)}
