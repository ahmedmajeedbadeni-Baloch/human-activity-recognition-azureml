import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.tensorflow
import os
from azureml.core import Workspace, Experiment
import azureml
print("MLflow version:", mlflow.__version__)
print("AzureML version:", azureml.__version__)

# Alphabetical order matches sklearn's LabelEncoder mapping produced in
# preprocess_data (fit on the "Activity" column, which sorts labels
# alphabetically by default).
ACTIVITY_LABELS = [
    "LAYING",
    "SITTING",
    "STANDING",
    "WALKING",
    "WALKING_DOWNSTAIRS",
    "WALKING_UPSTAIRS",
]


def init_mlflow():
    """Initialize MLflow tracking with Azure ML workspace"""
    try:
        ws = Workspace.from_config()
        mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
        experiment = Experiment(workspace=ws, name="Com774-Ahmed")  # Match your experiment name
        mlflow.set_experiment(experiment.name)
        print("MLflow tracking initialized successfully")
    except Exception as e:
        print(f"Error initializing MLflow: {str(e)}")
        raise


def preprocess_data(training_data_path):
    """
    Load the raw HAR dataset, split into train/test, encode the Activity
    label, and scale the sensor features to [0, 1].

    This mirrors the preprocessing verified in notebook/data_exploration.ipynb:
    the last two columns of the raw file are "subject" and "Activity", so the
    feature matrix is everything except those two columns.

    Returns the feature column names alongside the arrays so downstream
    steps (permutation importance) can label results by feature name instead
    of column index.
    """
    dataset = pd.read_csv(training_data_path)

    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    feature_names = train_data.columns[:-2].tolist()

    x_train, y_train = train_data.iloc[:, :-2], train_data.iloc[:, -1]
    x_test, y_test = test_data.iloc[:, :-2], test_data.iloc[:, -1]

    # Fit the encoder on the training labels only, then apply that same
    # mapping to the test labels. The original notebook called
    # fit_transform() on both, which happened to produce an identical
    # mapping for this dataset (all 6 classes appear in both splits and
    # LabelEncoder sorts alphabetically) but is not safe in general.
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test, feature_names


def build_model(input_dim, num_classes):
    """
    Build the Sequential network verified in notebook/data_exploration.ipynb:
    a single 64-unit hidden layer with sigmoid activation, 20% dropout, and a
    softmax output layer.
    """
    model = Sequential()
    model.add(Dense(units=64, kernel_initializer='normal', activation='sigmoid', input_dim=input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(units=num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_metrics(history, run):
    """
    Plot training and validation accuracy/loss over epochs.
    """
    plt.figure(figsize=(14, 6))

    # Plot Training vs Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Plot Training vs Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()

    # Save plot to file and log to MLflow
    plot_path = os.path.join(os.getcwd(), 'training_metrics.png')
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()


def evaluate_and_log(model, x_test, y_test):
    """
    Compute a per-class classification report and confusion matrix on the
    held-out split, print them, and log both as MLflow artifacts. Accuracy
    alone doesn't show whether the model is weak on any specific activity
    (sitting vs standing is a known hard pair in this dataset); this does.
    """
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    report = classification_report(y_test, y_pred, target_names=ACTIVITY_LABELS)
    print("Classification Report:\n" + report)

    report_path = os.path.join(os.getcwd(), "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ACTIVITY_LABELS, yticklabels=ACTIVITY_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(os.getcwd(), "confusion_matrix.png")
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()


def _keras_accuracy_scorer(estimator, X, y):
    """Scorer wrapper so sklearn's permutation_importance can call a Keras
    model (which doesn't expose the .score() method sklearn estimators do)."""
    preds = np.argmax(estimator.predict(X, verbose=0), axis=1)
    return accuracy_score(y, preds)


def log_permutation_importance(model, x_test, y_test, feature_names, top_n=15):
    """
    Rank the top_n most important input features by how much shuffling each
    one degrades held-out accuracy. This is a lightweight interpretability
    signal that needs no extra dependency beyond scikit-learn, as opposed to
    a library like SHAP which would need to be added separately.
    """
    result = permutation_importance(
        model, x_test, y_test,
        scoring=_keras_accuracy_scorer,
        n_repeats=5,
        random_state=42,
    )

    importances = pd.Series(result.importances_mean, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    top_features.iloc[::-1].plot(kind="barh")
    plt.xlabel("Mean accuracy drop when shuffled")
    plt.title(f"Top {top_n} features by permutation importance")
    plt.tight_layout()
    importance_path = os.path.join(os.getcwd(), "permutation_importance.png")
    plt.savefig(importance_path)
    mlflow.log_artifact(importance_path)
    plt.close()

    print("Top features by permutation importance:")
    print(top_features)


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True, help="Path to the input dataset CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
    args = parser.parse_args()

    # Initialize MLflow
    init_mlflow()

    # Start MLflow run
    with mlflow.start_run() as run:
        try:
            # Enable auto-logging
            mlflow.tensorflow.autolog(log_models=True)

            # Preprocess the dataset
            x_train, y_train, x_test, y_test, feature_names = preprocess_data(args.training_data)

            # Build the model
            model = build_model(input_dim=x_train.shape[1], num_classes=6)

            os.makedirs(args.output_dir, exist_ok=True)
            best_checkpoint_path = os.path.join(args.output_dir, "best_model.keras")

            callbacks = [
                # Stop training once val_loss stops improving for 3 epochs
                # in a row, and restore the weights from the best epoch
                # rather than whatever epoch training happened to end on.
                EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                # Separately persist the best-val_loss epoch's weights to
                # disk as training runs, in case the job is interrupted
                # before EarlyStopping/fit() returns.
                ModelCheckpoint(best_checkpoint_path, monitor="val_loss", save_best_only=True),
            ]

            history = model.fit(
                x_train, y_train,
                batch_size=64,
                epochs=20,
                validation_data=(x_test, y_test),
                callbacks=callbacks
            )

            # Save the trained model
            model_path = os.path.join(args.output_dir, "model")
            model.save(model_path)
            print(f"Model saved at {model_path}")
            print(f"Best checkpoint (lowest val_loss) saved at {best_checkpoint_path}")

            # Plot and save training metrics
            plot_metrics(history, run)

            # Per-class evaluation and interpretability
            evaluate_and_log(model, x_test, y_test)
            log_permutation_importance(model, x_test, y_test, feature_names)

            # Log final metrics
            mlflow.log_metrics({
                "final_training_accuracy": history.history['accuracy'][-1],
                "final_validation_accuracy": history.history['val_accuracy'][-1],
                "final_training_loss": history.history['loss'][-1],
                "final_validation_loss": history.history['val_loss'][-1]
            })

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise


if __name__ == "__main__":
    main()
