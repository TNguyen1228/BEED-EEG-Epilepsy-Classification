# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An end-to-end MLOps project that classifies epileptic seizures on the BEED (Bangalore EEG Epilepsy Dataset) with a scikit-learn Random Forest. It has four classes: 0 Healthy, 1 Generalized Seizure, 2 Focal Seizure, 3 Seizure Event (Mimic). The README, docstrings and comments are mostly in Vietnamese; follow that convention when editing existing prose.

## Commands

Run everything from the repo root. `load_config()` defaults to the relative path `config/config.yaml`, and the data, model and MLflow paths in the config are also relative.

```bash
pip install -r requirements.txt
python -m src.models.train                  # train, overwrite models/*.joblib + metrics.json, log to MLflow
pytest tests/ -v                            # full test suite
pytest tests/test_model.py::TestModelModule::test_preprocessing_pipeline   # single test
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload                    # API (Swagger at /docs)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
flake8 src api tests && black --check src api tests                        # lint (make lint)
docker compose up --build -d                # API :8000 + MLflow UI :5000
```

`make` targets wrap these commands, but `make` may not be installed on Windows, so run the commands directly.

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml), Python 3.10) fails only on flake8 errors `E9,F63,F7,F82`, which are syntax errors and undefined names. All other flake8 warnings are non-blocking, and CI does not run black. CI then runs pytest, runs training, and checks that both model artifacts exist.

## Architecture

**Config-driven.** [config/config.yaml](config/config.yaml) is the single source for the data path, the feature column names `X1`–`X16`, the label map, Random Forest hyperparameters, artifact paths and the MLflow tracking URI. The training code and the API both read it through `src.data.data_loader.load_config`.

**One sklearn Pipeline is the whole model.** [src/models/train.py](src/models/train.py) builds `Pipeline([("preprocessor", ...), ("classifier", RandomForestClassifier)])`. The preprocessor comes from `build_preprocessing_pipeline()` in [src/features/feature_engineering.py](src/features/feature_engineering.py) and is the custom transformer `EEGSpectralAndSpatialFeatureExtractor` followed by `RobustScaler`. The whole pipeline is saved with joblib. Because of that, the API passes **raw 16-channel values** (as a DataFrame with the config column names) to `MODEL.predict` and does no feature engineering of its own.

**The 16 values in a row are most likely 16 consecutive time samples of one channel, but this is inferred, not confirmed.** The extractor runs `np.fft.rfft(..., axis=1)` across the 16 values in a row, producing 9 bins. The docstrings and the README now use this temporal reading. The class name, the `include_spatial_spectrum` parameter and the pipeline step name still say "spatial" from an earlier reading that treated the values as 16 electrodes at one instant. Keep those names, because the pickled model depends on them. The public BEED description calls X1–X16 "channels", but the data points the other way:
- The correlation between two columns depends almost only on |i−j|, falling from 0.87 at distance 1 to about 0 at distance 6–9.
- All 16 columns have nearly the same standard deviation, about 37.

That is how 16 consecutive samples of one stationary signal behave, which at 256 Hz is about 62 ms. Under that reading, the FFT is a coarse time-domain PSD with 16 Hz bins, and `np.diff` is a time derivative rather than a spatial gradient. Rows are independent of each other: the correlation between row t and row t+1 is about 0.09.

The two readings use the same computation, so the model is unaffected either way. Do not build on either reading without confirming it, for example by adding Hz-based band features (which assume the temporal reading) or reordering columns (which matters under the spatial reading).

The extractor outputs the raw values, the PSD bins, total power, spectral entropy, spectral centroid, the high/low ratio, the diffs between neighboring values, and the per-row statistics.

The CSV also differs from the public BEED description. It has 8000 rows with 4 balanced classes of 2000 each, while the description says 16,000 segments and a binary label.

The dataset authors' paper, Najmusseher & Banu, "Feature Engineering for Epileptic Seizure Classification Using SeqBoostNet" (IJCDS 2025, doi:10.12785/ijcds/1571020131), does not settle the question. It describes "raw waveform signals from 16 EEG channels" at 256 Hz, but it never says what one row is or how the 20-second recordings were cut into rows. It does process BEED (4000×16) exactly like the UCI BONN CSV (4600×178), and the 178 `X1…X178` columns of the BONN CSV are consecutive time samples. That parallel supports the temporal reading.

**The committed model artifact is a pickle.** `models/random_forest_pipeline.joblib` and `models/metrics.json` are tracked in git, and `python -m src.models.train` overwrites both. The pickle refers to `src.features.feature_engineering.EEGSpectralAndSpatialFeatureExtractor` by its import path. If you rename or move that class or module, or change the shape of its output, you must retrain and commit the new artifacts, or the API will fail to load or predict. The Dockerfile also retrains when the image is built.

**MLflow is optional at runtime.** `train.py` guards the import with a try/except and saves the model locally either way. With MLflow present, each run logs params and metrics and registers the model as `BEED_EEG_RandomForest` in `sqlite:///mlflow.db`. There is a fallback `log_model` call for MLflow versions that need `skops_trusted_types`. `mlflow.db` and `mlruns/` are gitignored.

**API** ([api/app.py](api/app.py)). The FastAPI `lifespan` hook loads the config, the labels and the joblib model into module globals (`MODEL`, `CONFIG`, `LABELS`). If the model file is missing, the API still starts, and the predict endpoints return 503. Pydantic schemas in [api/schemas.py](api/schemas.py) enforce exactly 16 features for `/predict`. `/predict/batch` checks each sample's length by hand.

## Tests

Tests are written as `unittest.TestCase` classes and run through pytest; `pytest.ini` sets `pythonpath = .`.
- `test_data.py` reads the real `BEED_Data.csv`, which has 8000 rows.
- `test_model.py` trains small pipelines on random data.
- `test_api.py` creates `TestClient(app)` without a `with` block, so the lifespan hook never runs and no model is loaded. The tests cover only `/health`, `/info` and request validation (422). If a new test needs predictions, use `with TestClient(app) as client:` so the model loads.
