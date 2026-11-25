
import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/practice_dataset.csv"
MODEL_DIR = "model"
MLFLOW_DB = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "BotSentinel"

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "sport",
    "dport",
    "pkts",
    "bytes",
    "dur",
    "mean",
    "stddev",
    "sum",
    "min",
    "max",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "srate",
    "drate"
]

CATEGORICAL_FEATURES = [
    "flgs",
    "proto",
    "state"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "attack"


# ============================================================
# LOAD DATASET
# ============================================================

def load_data(data_path):

    print("\nLoading dataset...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(
        data_path,
        low_memory=False
    )

    # Remove columns that are entirely empty
    df = df.dropna(axis=1, how="all")

    required_columns = ALL_FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert numeric columns safely
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Convert categorical columns to strings
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string")

    # Convert target to numeric
    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    # Remove rows with invalid target values
    df = df.dropna(subset=[TARGET])

    # Keep only binary target values
    df = df[df[TARGET].isin([0, 1])]

    df[TARGET] = df[TARGET].astype(int)

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "Dataset must contain both classes: 0 and 1."
        )

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    print(f"Rows: {len(df)}")
    print(f"Features: {len(ALL_FEATURES)}")

    print("\nClass distribution:")
    print(y.value_counts())

    print("\nClass percentages:")
    print(
        (y.value_counts(normalize=True) * 100).round(4)
    )

    return X, y


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

def get_model_configurations():

    return [

        {
            "name": "LogisticRegression_C0.1",
            "model": LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=RANDOM_STATE
            )
        },

        {
            "name": "LogisticRegression_C1",
            "model": LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=RANDOM_STATE
            )
        },

        {
            "name": "RandomForest",
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        }

    ]


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test, predictions
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),

        "precision": precision_score(
            y_test, predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test, predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_test, predictions,
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        ),

        "pr_auc": average_precision_score(
            y_test, probabilities
        )
    }

    print("\nMetrics:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0
        )
    )

    return metrics


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    global DATA_PATH

    parser = argparse.ArgumentParser(
        description="BotSentinel model training"
    )

    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help="Path to the dataset CSV"
    )

    args = parser.parse_args()

    DATA_PATH = args.data

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    X, y = load_data(DATA_PATH)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_DB)

    mlflow.set_experiment(EXPERIMENT_NAME)

    configurations = get_model_configurations()

    results = []

    best_f1 = -1
    best_model = None
    best_model_name = None

    # Train each configuration
    for config in configurations:

        model_name = config["name"]
        classifier = config["model"]

        print("\n" + "=" * 55)
        print(f"Training: {model_name}")
        print("=" * 55)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier)
            ]
        )

        with mlflow.start_run(run_name=model_name):

            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test
            )

            # Log model configuration
            mlflow.log_param(
                "model_name",
                model_name
            )

            mlflow.log_param(
                "training_rows",
                len(X_train)
            )

            mlflow.log_param(
                "testing_rows",
                len(X_test)
            )

            mlflow.log_param(
                "feature_count",
                len(ALL_FEATURES)
            )

            mlflow.log_param(
                "random_state",
                RANDOM_STATE
            )

            # Log evaluation metrics
            mlflow.log_metrics(metrics)

            # Use pickle serialization for our own locally
            # trained models to avoid skops type-trust errors.
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="pickle"
            )

            results.append({
                "model": model_name,
                **metrics
            })

            # Select best model using F1
            if metrics["f1"] > best_f1:

                best_f1 = metrics["f1"]
                best_model = pipeline
                best_model_name = model_name

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if best_model is None:
        raise RuntimeError(
            "Training failed: no model was successfully trained."
        )

    model_path = os.path.join(
        MODEL_DIR,
        "model.joblib"
    )

    joblib.dump(best_model, model_path)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)

    print(f"Best model: {best_model_name}")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Saved model: {model_path}")

    # ========================================================
    # RESULTS COMPARISON
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\nMODEL COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results_path = os.path.join(
        MODEL_DIR,
        "training_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
