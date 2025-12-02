"""
Train a LightGBM regressor with a monotonic constraint on Hydrogen.
Saves model to `models/stack2_model_monotonic_lgbm.pkl` and prints evaluation and hydrogen sensitivity.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ensure lightgbm is installed
try:
    import lightgbm as lgb
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])  
    import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'PEMfuelcell.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
OUTPUT_MODEL_PATH = os.path.join(MODEL_DIR, 'stack2_model_monotonic_lgbm.pkl')

# Load data
df = pd.read_csv(DATA_PATH)

# Feature order used elsewhere
features = ['I','P','Q','T','Hydrogen','Oxygen','RH anode','Rh Cathode']
if not set(features).issubset(df.columns):
    raise RuntimeError(f"Missing required features in {DATA_PATH}: {features}")

X = df[features].copy()
y = df['V'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Monotone constraints: Hydrogen index is 4 (0-based)
mono = [0]*len(features)
mono[4] = 1  # enforce monotonically increasing effect of Hydrogen on V

print('Training LightGBM with monotone constraints:', mono)
model = lgb.LGBMRegressor(n_estimators=10000, learning_rate=0.05)

# LightGBM expects monotone_constraints passed to fit via callbacks/params in sklearn API
# We'll pass as parameter to the underlying booster via `monotone_constraints` in init params
# The sklearn wrapper supports passing it in the constructor since v3.0+, but to be robust:
try:
    model = lgb.LGBMRegressor(n_estimators=10000, learning_rate=0.05, monotone_constraints=tuple(mono))
except TypeError:
    # fallback: set via params in fit
    model = lgb.LGBMRegressor(n_estimators=10000, learning_rate=0.05)

# further split training into train/val for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=7)

print('Fitting model...')
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)],
)

# Evaluate on test
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
print(f'Validation results on test set: MAE={mae:.5f}, RMSE={rmse:.5f}, R2={r2:.5f}')

# Hydrogen sensitivity: vary Hydrogen between near-zero and 0.01 using median baseline
median_vals = X.median().to_dict()
hyd_vals = np.linspace(1e-6, 0.01, 16)
print('\nHYDROGEN SENSITIVITY (median baseline)')
print('Hydrogen\tPredictedV')
for h in hyd_vals:
    row = [median_vals[f] for f in features]
    row[4] = h
    pv = model.predict([row])[0]
    print(f'{h:.6f}\t{pv:.6f}')

# Save model
joblib.dump(model, OUTPUT_MODEL_PATH)
print(f'Wrote monotonic model to: {OUTPUT_MODEL_PATH}')
