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

# 1. Load RAW Data (Critical: Do not load the pre-scaled file)
df = pd.read_csv(r"D:\Coding\Major-Project\new_\data\PEMfuelcell.csv")

# 2. Select Features (Critical: No P, No Q)
X = df[["I", "T", "Hydrogen", "Oxygen", "RH anode", "Rh Cathode"]].values
y = df["V"].values

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define Learners (SAME models, BETTER settings)
base_learners_constrained = [
    # ElasticNet: Kept as is (It's weak, but fine to include)
    ("enet", make_pipeline(StandardScaler(), ElasticNet(max_iter=5000))),
    
    # SVR: Increased C=100 (Tightens the fit)
    ("svr",  make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, epsilon=0.1))),
    
    # KNN: Kept as is
    ("knn",  make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7))),
    
    # GBR: THE HERO FIX -> Changed max_depth=3 to max_depth=10
    ("gbr",  GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=42)),
]

# Meta Learner (Same as yours)
meta = RidgeCV(alphas=np.logspace(-4, 4, 25))

# Stacking Model (Same structure)
stack_model = StackingRegressor(
    estimators=base_learners_constrained,
    final_estimator=meta,
    cv=5,
    passthrough=False,
    n_jobs=-1
)

# 5. Train
print("Training Optimized Stacking Model...")
stack_model.fit(X_train, y_train)

# 6. Evaluate
y_pred = stack_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nFinal Results:")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# 7. Test 66A Case
test_row = np.array([[41.77525263, 59.22875598, 0.002981592, 0.015133534, 1.000473951, 1.000004215]])
pred_val = stack_model.predict(test_row)[0]
print(f"\nPrediction for 41.75A: {pred_val:.4f} V")

import joblib

# Save the model to a file
joblib.dump(stack_model, 'raw_stack_model.pkl')
print("Model saved as raw_stack_model.pkl")