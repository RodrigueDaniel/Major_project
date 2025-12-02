"""
Compare original stacking model and monotonic LightGBM on the same test split.
Prints MAE, RMSE, R2 and correlation between Hydrogen and predictions.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'PEMfuelcell.csv')
MODELS_DIR = os.path.join(ROOT, 'models')

# Paths
stack_model_path = os.path.join(MODELS_DIR, 'stack2_model.pkl')
stack_raw_wrapper = os.path.join(MODELS_DIR, 'stack2_model_raw_input.pkl')
mono_path = os.path.join(MODELS_DIR, 'stack2_model_monotonic_lgbm.pkl')

# Load data
df = pd.read_csv(DATA_PATH)
features = ['I','P','Q','T','Hydrogen','Oxygen','RH anode','Rh Cathode']
X = df[features]
y = df['V'].values

# Use same split as monotonic script: random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {}
# Try to load stacking model; prefer the raw-input wrapper if available
if os.path.exists(stack_raw_wrapper):
    try:
        models['stacking'] = joblib.load(stack_raw_wrapper)
    except Exception:
        models['stacking'] = joblib.load(stack_model_path)
else:
    models['stacking'] = joblib.load(stack_model_path)

# Load monotonic
models['monotonic_lgbm'] = joblib.load(mono_path)

# Evaluate
print('Evaluation on same test set (random_state=42)')
for name, m in models.items():
    # Some models (LGBM) accept DataFrame directly; others too
    y_pred = m.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(X_test['Hydrogen'].values, y_pred)[0,1]
    print(f"\nModel: {name}")
    print(f" MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}, Corr(Hydrogen, pred)={corr:.6f}")

# Quick check: mean absolute difference between model predictions
if 'stacking' in models and 'monotonic_lgbm' in models:
    diff = np.mean(np.abs(models['stacking'].predict(X_test) - models['monotonic_lgbm'].predict(X_test)))
    print(f"\nMean absolute difference between stacking and monotonic predictions on test set: {diff:.6f}")

print('\nDone')
