# Housing-MLE Project: Complete Workflow

## Phase 1: Data Exploration & Model Selection (Notebooks)

### Notebook 00 - Data Split
Input: Raw data (housets_original.csv + us_metros.csv)
Process: Temporal split (train<2020, eval 2020-2022, holdout>=2022)
Output: train_raw.csv, eval_raw.csv, holdout_raw.csv
Status: COMPLETED

### Notebook 01 - EDA & Cleaning
Input: Raw data (3 CSVs)
Process: Deduplication, outlier removal, city to lat/lng mapping
Output: train_clean.csv, eval_clean.csv, holdout_clean.csv
Status: COMPLETED

### Notebook 02 - Feature Engineering
Input: Clean data (3 CSVs)
Process: Time features (year, quarter, month), encoding (zipcode→freq, city→target_enc)
Output: train_final.csv, eval_final.csv, holdout_final.csv
Status: COMPLETED

### Notebook 03 - Baseline Model
Input: Final data (3 CSVs)
Process: DummyRegressor (predict mean price)
Output: baseline_metrics.csv
Key Result: MAE Eval = 187,105 USD (reference to beat)
Status: COMPLETED

### Notebook 04 - Linear Models
Input: Final data (3 CSVs)
Process: OLS, Ridge (L2), Lasso (L1) with hyperparameter tuning
Output: linear_models_eval.csv, linear_models_holdout.csv
Key Result: MAE Eval = 62,128 USD (3.3x better than baseline)
Status: COMPLETED

### Notebook 05 - XGBoost
Input: Final data (3 CSVs)
Process: Train XGBoost with standard hyperparameters
Output: xgboost_model.pkl, xgboost_metrics.csv, xgboost_feature_importance.csv
Key Result: Compare with linear models, extract feature importance
Status: READY TO RUN

### Notebook 06 - Hyperparameter Tuning + MLflow
Input: Final data (3 CSVs)
Process: Optuna optimization (20 trials), log results in MLflow
Output: xgboost_best_model.pkl, xgboost_best_metrics.csv, MLflow tracking
Key Result: Best hyperparameters and model artifact
Status: READY TO RUN

---

## Phase 2: Production Code (src/)

### Structure
src/
├── feature_pipeline/
│   ├── __init__.py
│   └── feature_pipeline.py          (Translate notebooks 01-02)
├── training_pipeline/
│   ├── __init__.py
│   └── train.py                      (Translate notebook 06)
├── inference_pipeline/
│   ├── __init__.py
│   └── inference.py                  (Batch/real-time predictions)
└── api/
    ├── __init__.py
    └── main.py                       (FastAPI endpoints)

### Key Principles
1. Load data with relative paths (not hardcoded)
2. Use dataclasses or Pydantic for input/output validation
3. Add logging for debugging
4. No data leakage: encodings fitted on train, applied to eval/holdout
5. Test coverage for each module

---

## Phase 3: Testing & Configuration

### tests/
- test_features.py: Validate feature_pipeline output
- test_training.py: Validate model training
- test_inference.py: Validate predictions

### configs/
- app_config.yml: Feature engineering params (smoothing_factor, outlier_threshold)
- mlflow_config.yml: MLflow tracking URI, experiment name
- ge_expectations.yml: Great Expectations validation rules

---

## Phase 4: Containerization & Deployment

### Docker
- Dockerfile: Package API backend
- Dockerfile.streamlit: Package UI frontend

### CI/CD
- .github/workflows/ci.yml: GitHub Actions pipeline
  * Run tests
  * Build Docker images
  * Push to AWS ECR
  * Deploy to AWS ECS

---

## Performance Summary

| Model | MAE (Eval) | RMSE (Eval) | R2 (Eval) | Notes |
|-------|-----------|-----------|---------|-------|
| Baseline | 187,105 | 270,576 | -0.14 | Dummy (predict mean) |
| Linear (OLS) | 62,128 | 96,061 | 0.86 | Good baseline model |
| Linear (Ridge) | 62,146 | 96,043 | 0.86 | Slight regularization |
| XGBoost (std) | TBD | TBD | TBD | Running notebook 05 |
| XGBoost (tuned) | TBD | TBD | TBD | Running notebook 06 |

---

## Environment & Tools

- Language: Python 3.11
- Package Manager: uv
- ML Libraries: pandas, scikit-learn, XGBoost, Optuna, MLflow
- API: FastAPI, Streamlit
- Container: Docker
- Cloud: AWS (S3, ECR, ECS)
- CI/CD: GitHub Actions

---

## Execution Order

### Quick Path (for slow machines)
1. Run notebooks 00-02 once (data prep)
2. Run notebook 03 (baseline)
3. Run notebook 04 (linear models)
4. Run notebook 05 (XGBoost)
5. Run notebook 06 (tuning - can reduce n_trials if needed)

### Optimization Tips
- Reduce data sample if running out of memory
- Reduce n_trials in Optuna (e.g., 20→10)
- Use max_depth smaller (e.g., 6→4)
- Increase learning_rate (faster convergence)

---

## Next Steps

After Phase 1 (notebooks complete):
1. Create src/feature_pipeline/feature_pipeline.py from notebooks 01-02
2. Create src/training_pipeline/train.py from notebook 06
3. Write tests in tests/ directory
4. Create configuration files in configs/
5. Build Docker images
6. Setup CI/CD pipeline
