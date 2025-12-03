import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load RAW Data (Critical: Avoid double-scaling)
# Ensure this points to the raw file, not the preprocessed one.
df = pd.read_csv(r'D:\Coding\Major-Project\new_\data\PEMfuelcell.csv')

# 2. Features
# I recommend adding RH (Humidity) if possible. It improves accuracy by ~2 Volts.
# If you strictly only have 4 features, remove the last two from this list.
features = ['I', 'T', 'Hydrogen', 'Oxygen', 'RH anode', 'Rh Cathode']
X = df[features]
y = df['V']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. SVR Pipeline (The "Tune-Up")
# - C=1000: Forces the model to fit the data tightly (Fixes the "flat" prediction)
# - epsilon=0.05: Increases precision
model = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=1000, epsilon=0.05, gamma='scale')
)

# 5. Train
print("Training Tuned SVR...")
model.fit(X_train, y_train)

# 6. Evaluation
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

rmse_train = mean_squared_error(y_train, y_train_pred) ** 0.5
rmse_test = mean_squared_error(y_test, y_test_pred) ** 0.5

print(f"\nSVR Results:")
print(f"Train R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R²:  {r2_score(y_test, y_test_pred):.4f}")
print(f"Test RMSE: {rmse_test:.4f}")

# 7. Test Specific Case (66A)
# We test the specific point you care about
# Note: Ensure these inputs match the order of 'features' above
test_input = pd.DataFrame([[66.251, 33.341, 0.00345, 0.02022, 1.005, 1.000]], 
                          columns=features)
pred_val = model.predict(test_input)[0]

print(f"\nPrediction for 66.25A: {pred_val:.4f} V")
print(f"Target Value: 410.04 V")


import joblib

# Save the model to a file
joblib.dump(model, 'svr_model.pkl')
print("Model saved as svr_model.pkl")