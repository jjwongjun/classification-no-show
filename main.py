# Standard library imports
import argparse
import logging
import os
import sys
import warnings

# Third-party imports
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

# Local application/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

REQUIRED_CONFIG_KEYS = [
    "file_path", "target_column", "val_test_size", "val_size",
    "logreg_param_grid", "knn_param_grid", "dt_param_grid", "cv", "scoring",
    "numerical_features", "nominal_features", "ordinal_features", "passthrough_features",
    "room_categories", "num_adults_categories", "num_children_categories",
]


def setup_logging(log_file: str = None) -> None:
    """
    Configure logging to the console and, optionally, to a file with timestamps and levels.

    Args:
        log_file (str): Path of the log file. If None, only console logging is used.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def load_config(config_path: str) -> dict:
    """
    Load the YAML configuration file and check that it is valid.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The validated configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML cannot be parsed, keys are missing or values are out of range.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}. Use --config to set its path.")
    except yaml.YAMLError as e:
        raise ValueError(f"Config file {config_path} is not valid YAML: {e}")

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    for key in ["val_test_size", "val_size"]:
        if not 0 < config[key] < 1:
            raise ValueError(f"Config '{key}' must be between 0 and 1, got {config[key]}")
    if config["cv"] < 2:
        raise ValueError(f"Config 'cv' must be at least 2, got {config['cv']}")
    if config.get("class_weight") not in (None, "balanced"):
        raise ValueError("Config 'class_weight' must be 'balanced' or null.")
    return config


def main():
    parser = argparse.ArgumentParser(description="Hotel no-show classification pipeline")
    parser.add_argument(
        "--config", default="./src/config.yaml", help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    # Hide only sklearn convergence warnings (not every warning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    try:
        config = load_config(args.config)
        setup_logging(config.get("log_file"))

        # Load CSV file into a DataFrame
        try:
            df = pd.read_csv(config["file_path"])
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {config['file_path']}")

        # Initialize and run data preparation
        data_prep = DataPreparation(config)
        cleaned_df = data_prep.clean_data(df)

        # Initialize model training with the created preprocessor
        model_training = ModelTraining(config, data_prep.preprocessor)

        # Split the data
        X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(
            cleaned_df
        )

        # Train and evaluate baseline models with default hyperparameters
        baseline_models, baseline_metrics = (
            model_training.train_and_evaluate_baseline_models(
                X_train, y_train, X_val, y_val
            )
        )

        # Train and evaluate tuned models with hyperparameter tuning
        tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
            X_train, y_train, X_val, y_val
        )

        # Combine all models and their metrics into dictionaries
        all_models = {**baseline_models, **tuned_models}
        all_metrics = {**baseline_metrics, **tuned_metrics}

        # Find the best model based on ROC AUC score
        best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["ROC AUC"])
        best_model = all_models[best_model_name]
        logging.info(f"Best Model Found: {best_model_name}")

        # Choose the decision threshold on the validation set, then lock it for the test set
        threshold = model_training.find_optimal_threshold(best_model, X_val, y_val)

        # Evaluate the best model on the test set
        final_metrics = model_training.evaluate_final_model(
            best_model, X_test, y_test, best_model_name, threshold=threshold
        )

        # Save the model, metrics, threshold and config snapshot
        model_training.save_artifacts(best_model, best_model_name, final_metrics, threshold)

    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Pipeline stopped: {e}")
        sys.exit(1)
    except Exception:
        logging.exception("Unexpected error in pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
