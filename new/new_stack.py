import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, RidgeCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Load Data
df = pd.read_csv(r"D:\Coding\Major-Project\new_\data\PEMfuelcell.csv")

# ==============================================================================
# THE ONLY CHANGE: Remove 'P' and 'Q' from inputs
# ==============================================================================
# Old Line: X = df[["I","P","Q","T","Hydrogen","Oxygen","RH anode","Rh Cathode"]].values
# New Line:
X = df[["I", "T", "Hydrogen", "Oxygen", "RH anode", "Rh Cathode"]].values
y = df["V"].values

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define the Stacking Model (EXACTLY as you had it)
base_learners_baseline = [
    ("enet", make_pipeline(StandardScaler(), ElasticNet(max_iter=5000))),
    ("svr",  make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1))),
    ("knn",  make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7))),
    ("gbr",  GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)),
]

meta = RidgeCV(alphas=np.logspace(-4, 4, 25))

stack_baseline = StackingRegressor(
    estimators=base_learners_baseline,
    final_estimator=meta,
    cv=5,
    passthrough=False
)

# 4. Train
print("Training Stacking Model (Pure Physics)...")
stack_baseline.fit(X_train, y_train)

# 5. Evaluate
y_pred_base = stack_baseline.predict(X_test)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
r2_base = r2_score(y_test, y_pred_base)
mae_base = mean_absolute_error(y_test, y_pred_base)

print("\n===== Baseline Stack Results (Without P/Q) =====")
print(f"RMSE: {rmse_base:.4f}")
print(f"R²:   {r2_base:.4f}")
print(f"MAE:  {mae_base:.4f}")

# 6. Test Your Specific Case (66A)
# Note: Input has only 6 values now
# 41.77525263,424.2315095,17.72237848,3.251180575,59.22875598,0.002981592,0.015133534,1.000473951,1.000004215
test_row = np.array([[41.77525263, 59.22875598, 0.002981592, 0.015133534, 1.000473951, 1.000004215]])
pred_val = stack_baseline.predict(test_row)[0]
print(f"\nPrediction for 41.775A: {pred_val:.4f} V")

import joblib

# Save the model to a file
joblib.dump(stack_baseline, 'raw_data_stack_model.pkl')
print("Model saved as raw_data_stack_model.pkl")