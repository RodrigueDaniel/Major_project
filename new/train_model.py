import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# CONFIGURATION
# ==========================================
# We explicitly EXCLUDE 'P' and 'Q' (Power/Heat) because they are answers, not inputs.
FEATURES = ['I', 'T', 'Hydrogen', 'Oxygen', 'RH anode', 'Rh Cathode']
TARGET = 'V'
RAW_DATA_PATH = r'D:\Coding\Major-Project\new_\data\PEMfuelcell.csv'
MODEL_OUTPUT_PATH = 'pemfc_model.pkl'

def train():
    print(f"Loading data from {RAW_DATA_PATH}...")
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find '{RAW_DATA_PATH}'. Please ensure it is in this folder.")
        return

    # Prepare Feature Matrix (X) and Target (y)
    # This automatically drops P and Q by only selecting the physics columns.
    try:
        X = df[FEATURES]
        y = df[TARGET]
    except KeyError as e:
        print(f"Error: Your CSV is missing required columns. Missing: {e}")
        return

    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Machine Learning Pipeline
    # 1. Imputer: Fills any missing gaps in data
    # 2. Scaler: Normalizes values (helps math)
    # 3. Model: Random Forest (Best for complex fuel cell physics)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))
    ])

    print("Training the model (this learns the physics)...")
    pipeline.fit(X_train, y_train)

    # Evaluate Performance
    print("Validating model accuracy...")
    y_pred = pipeline.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    # older scikit-learn versions don't accept the `squared` kwarg
    # compute RMSE manually for compatibility
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print("-" * 30)
    print(f"TRAINING COMPLETE")
    print(f"Accuracy (R2 Score): {accuracy:.4f} (Perfect is 1.0)")
    print(f"Avg Error (RMSE):    {rmse:.4f} Volts")
    print("-" * 30)

    # Save the Pipeline
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"Model saved to '{MODEL_OUTPUT_PATH}'")
    print("You can now use 'predict_voltage.py' to make predictions!")

if __name__ == '__main__':
    train()