Machine Failure Prediction — Team ZBH LOOP
DATA DIVERGENCE Hackathon — Track 1 (Industrial Intelligence / Smart Manufacturing)
Predicts whether a machine will fail based on sensor readings, using a blended GradientBoosting + RandomForest classifier.
How it works
Input features (6): `Type`, `Air temperature \[K]`, `Process temperature \[K]`, `Rotational speed \[rpm]`, `Torque \[Nm]`, `Tool wear \[min]`
Preprocessing (all inside the saved pipeline):
Median imputation + scaling for numeric columns
Most-frequent imputation + one-hot encoding for `Type`
Engineered features: temperature gap (process − air), torque × speed, torque × tool wear
Model: Soft-voting ensemble of `GradientBoostingClassifier` + `RandomForestClassifier` (300 trees)
Threshold: Not fixed at 0.5 — tuned via 5-fold stratified cross-validation to maximize F1 (found via precision-recall curve), then saved inside the model file
Output: `failure\_probability`, `predicted\_failure` (0/1), `prediction\_label` (Healthy / Potential failure)
Everything (preprocessing + model + threshold) is bundled into one `.joblib` file, so predicting on new data doesn't require re-running any feature engineering manually.
Requirements
```bash
pip install scikit-learn pandas numpy joblib
```
Commands
Train (uses `train.csv`, produces `machine\_failure\_model.joblib`):
```bash
python model.py train
```
Runs 5-fold cross-validation and prints PR-AUC, F1, precision, recall, and the chosen threshold. Takes a few minutes.
For a faster run (skips cross-validation, uses a pre-set threshold):
```bash
python model.py train --quick
```
Predict (on any CSV containing the 6 required columns):
```bash
python model.py predict your\_file.csv
```
Saves results to `predictions\_your\_file.csv` and prints them to the terminal.
Repo structure
ZBH_LOOP_Methodology_Report/
├── README.md                              # Project overview and usage (this file)
├── ZBH_LOOP_Methodology_Report (1).pdf    # Methodology report
└── files/
    ├── model.py                           # Train / predict script (CLI)
    └── machine_failure_model.joblib       # Saved pipeline: preprocessing + model + threshold
