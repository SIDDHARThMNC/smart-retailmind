import os
import shutil
import pytest
import pandas as pd

# absolute paths so tests work from any directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SAMPLE_DATA       = os.path.join(BASE_DIR, "data", "sample_retail_sales.csv")
CLEANED_DATA      = os.path.join(BASE_DIR, "data", "cleaned_retail_sales.csv")
MODEL_PATH        = os.path.join(BASE_DIR, "backend", "saved_models", "demand_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "backend", "saved_models", "preprocessor.pkl")

_cleaned_backup       = None
_model_existed        = False
_preprocessor_existed = False


def pytest_configure(config):
    global _cleaned_backup, _model_existed, _preprocessor_existed

    # set working directory to project root so relative paths work
    os.chdir(BASE_DIR)

    import sys
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    # back up existing cleaned data
    if os.path.exists(CLEANED_DATA):
        _cleaned_backup = pd.read_csv(CLEANED_DATA)

    _model_existed        = os.path.exists(MODEL_PATH)
    _preprocessor_existed = os.path.exists(PREPROCESSOR_PATH)

    # copy sample data so tests have something to work with
    if os.path.exists(SAMPLE_DATA):
        os.makedirs(os.path.dirname(CLEANED_DATA), exist_ok=True)
        shutil.copy(SAMPLE_DATA, CLEANED_DATA)

    # train model so prediction tests don't fail
    try:
        from backend.ml.training import train_models
        train_models(CLEANED_DATA)
        print("\n[conftest] Model trained successfully.")
    except Exception as e:
        print(f"\n[conftest] Model training failed: {e}")


def pytest_unconfigure(config):
    # restore cleaned data to what it was before tests
    if _cleaned_backup is not None:
        _cleaned_backup.to_csv(CLEANED_DATA, index=False)
    elif os.path.exists(CLEANED_DATA):
        os.remove(CLEANED_DATA)

    # remove model files only if they didn't exist before
    if not _model_existed and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if not _preprocessor_existed and os.path.exists(PREPROCESSOR_PATH):
        os.remove(PREPROCESSOR_PATH)
