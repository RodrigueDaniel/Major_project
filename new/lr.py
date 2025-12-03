import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline # <--- NEW IMPORT
import joblib

# Step 1: Load the Excel file
df = pd.read_csv(r'D:\Coding\Major-Project\new_\data\PEMfuelcell.csv')

# Step 2: Drop missing values
df.dropna(inplace=True)

# Step 3: Define features and target
features = ['I', 'T', 'Hydrogen', 'Oxygen', 'RH anode', 'Rh Cathode']
X = df[features]
y = df['V']

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === THE FIX: Use a Pipeline ===
# This bundles the Poly step, the Scaler, and the Model into one object
model_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Step 5 & 6: Train the Pipeline
# The pipeline automatically handles the poly transform and scaling internally
model_pipeline.fit(X_train, y_train)

# Step 7: Make predictions
# We pass X_test (raw data) directly; the pipeline handles the rest
y_pred = model_pipeline.predict(X_test)

# Step 8: Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance (Polynomial Pipeline Degree 3):")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Step 9: Test Specific Case
# NOTE: Your features list has 6 items, so your input must have 6 values.
# I added dummy 0.0 values for 'RH anode' and 'Rh Cathode' to prevent a crash.
# Please replace them with real values.
test_input = pd.DataFrame([[66.251, 33.341, 0.00345, 0.02022, 1.0, 1.0]], columns=features)

# Pipeline Magic: No need to manually poly_transform or scale here!
pred_val = model_pipeline.predict(test_input)[0]

print(f"\nPrediction for test case: {pred_val:.4f} V")

# # Step 10: Plot
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
# plt.xlabel('Actual Voltage')
# plt.ylabel('Predicted Voltage')
# plt.title('Actual vs Predicted Voltage (Polynomial Pipeline)')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# === FINAL STEP: SAVE AS ONE FILE ===
joblib.dump(model_pipeline, "pemfc_complete_model.pkl")

print("Success! Model, Scaler, and Poly features saved in 'pemfc_complete_model.pkl'")


import joblib
import pandas as pd

# 1. Load the single file
model = joblib.load("pemfc_complete_model.pkl")

# 2. Create raw data (Order must match the original 'features' list)
new_data = pd.DataFrame([[66.25, 33.34, 0.003, 0.02, 1.0, 1.0]], 
                        columns=['I', 'T', 'Hydrogen', 'Oxygen', 'RH anode', 'Rh Cathode'])

# 3. Predict directly (The pipeline does the poly/scaling automatically)
prediction = model.predict(new_data)
print(prediction)