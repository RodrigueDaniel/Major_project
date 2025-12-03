import os
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
# CHANGE 1: REMOVED 'P' and 'Q' (Model expects 6 features, not 8)
FEATURES = ['I', 'T', 'Hydrogen', 'Oxygen', 'RH anode', 'Rh Cathode']

# CHANGE 2: RESET FACTOR TO 1.0
# The new model is accurate. We only use 1.04 if we detect under-prediction later.
CALIBRATION_FACTOR = 1.0 

# Physical Limit: A 400-cell stack cannot physically exceed ~1.25V/cell (500V total)
MAX_THEORETICAL_VOLTAGE = 500.0


def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[column].clip(lower=lower_bound, upper=upper_bound)


def preprocess(df, skip_scaling=False):
    # Ensure feature columns exist
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Impute missing values (mean)
    imputer = SimpleImputer(strategy='mean')
    # Note: If target 'V' is missing from input (prediction mode), we handle it
    cols_to_transform = FEATURES + (["V"] if "V" in df.columns else [])
    
    df_imputed = pd.DataFrame(imputer.fit_transform(df[cols_to_transform]),
                              columns=cols_to_transform)

    # Cap outliers
    for col in df_imputed.columns:
        df_imputed[col] = cap_outliers(df_imputed, col)

    scaler = None
    if skip_scaling:
        # Do not scale, keep imputed & capped features as-is
        preprocessed_df = df_imputed[FEATURES].reset_index(drop=True)
        if 'V' in df_imputed.columns:
            preprocessed_df = pd.concat([preprocessed_df, df_imputed['V'].reset_index(drop=True)], axis=1)
        return preprocessed_df, imputer, scaler

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_imputed[FEATURES])
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)

    if 'V' in df_imputed.columns:
        preprocessed_df = pd.concat([X_scaled, df_imputed['V'].reset_index(drop=True)], axis=1)
    else:
        preprocessed_df = X_scaled

    return preprocessed_df, imputer, scaler


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict(model, X):
    return model.predict(X)


def main(raw_path, preprocessed_out, model_path, predictions_out, plot_out=None, show_plot=False, skip_scaling=False, save_preprocessed=False, overwrite_preprocessed=False):
    # Handle paths
    raw_path = os.path.abspath(raw_path) if raw_path else None
    preprocessed_out = os.path.abspath(preprocessed_out)
    model_path = os.path.abspath(model_path)
    predictions_out = os.path.abspath(predictions_out)

    # Load Data
    df_raw = None
    if raw_path and os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)

    if df_raw is None:
        raise FileNotFoundError(f"Raw data file not found: {raw_path} - or provide input rows via --row flag")

    # Preprocess
    # Note: For the new pipeline model, it handles scaling internally. 
    # But if your code uses external scaling, we keep this. 
    # Generally, for single-row prediction with a pipeline model, we use --skip-scaling.
    preprocessed_df, imputer, scaler = preprocess(df_raw, skip_scaling=skip_scaling)

    # Save preprocessed (Optional)
    if save_preprocessed:
        if os.path.exists(preprocessed_out) and not overwrite_preprocessed:
            print(f"Skipping save: '{preprocessed_out}' exists.")
        else:
            os.makedirs(os.path.dirname(preprocessed_out), exist_ok=True)
            preprocessed_df.to_csv(preprocessed_out, index=False)
            print(f"Preprocessed data saved to '{preprocessed_out}'")

    # Load model
    model = load_model(model_path)

    # Prepare inputs for prediction
    X_for_pred = preprocessed_df.drop(columns=['V'], errors='ignore')

    # 1. GET RAW PREDICTIONS
    # If the model is a Pipeline, it might expect unscaled data. 
    # If you used --skip-scaling, X_for_pred is unscaled (correct for Pipeline).
    raw_preds = predict(model, X_for_pred.values)

    # 2. APPLY CALIBRATION FIX (Default 1.0)
    calibrated_preds = raw_preds * CALIBRATION_FACTOR

    # 3. APPLY SAFETY CLAMP
    final_preds = np.minimum(calibrated_preds, MAX_THEORETICAL_VOLTAGE)

    # Print results
    try:
        if hasattr(final_preds, '__iter__'):
            preds_list = [float(p) for p in final_preds]
        else:
            preds_list = [float(final_preds)]
    except Exception:
        preds_list = list(final_preds if isinstance(final_preds, (list, tuple, np.ndarray)) else [final_preds])
    
    print("\n" + "="*40)
    print(" PREDICTION RESULTS")
    print("="*40)
    for i, val in enumerate(preds_list, start=1):
        print(f"  Row {i}: {val:.6f} V")

    # Save output
    df_out = df_raw.copy().reset_index(drop=True)
    df_out['V_pred'] = final_preds

    os.makedirs(os.path.dirname(predictions_out), exist_ok=True)
    df_out.to_csv(predictions_out, index=False)
    print(f"Predictions saved to '{predictions_out}'")

    # Plot
    try:
        if 'I' in df_out.columns and 'V_pred' in df_out.columns:
            df_plot = df_out.copy()
            df_plot_sorted = df_plot.sort_values('I')
            plt.figure(figsize=(8, 6))
            
            plt.scatter(df_plot['I'], df_plot['V_pred'], s=30, color='tab:blue', label='Predicted')
            
            if len(df_plot_sorted) > 1:
                plt.plot(df_plot_sorted['I'], df_plot_sorted['V_pred'], color='tab:orange', linewidth=1, label='Trend')
            else:
                single_I = float(df_plot_sorted['I'].iloc[0])
                single_V = float(df_plot_sorted['V_pred'].iloc[0])
                span = max(abs(single_I) * 0.05, 0.1)
                plt.plot([single_I - span, single_I + span], [single_V, single_V], color='tab:orange', linestyle='--', linewidth=1, label='Level')
                plt.annotate(f"{single_V:.2f} V", xy=(single_I, single_V), xytext=(5, 5), textcoords='offset points')

            plt.xlabel('Current (I)')
            plt.ylabel('Predicted Voltage (V)')
            plt.title('Predicted Voltage vs Current')
            plt.grid(True)
            plt.legend()
            
            if plot_out is None:
                plot_out = os.path.join(os.path.dirname(predictions_out), 'predictions_plot.png')
            os.makedirs(os.path.dirname(plot_out) or '.', exist_ok=True)
            plt.savefig(plot_out, dpi=200, bbox_inches='tight')
            print(f"Plot saved to '{plot_out}'")
            
            if show_plot:
                plt.show()
            plt.close()
    except Exception as e:
        print(f"Failed to generate plot: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict PEM fuel cell voltage (Pure Physics Model)')
    parser.add_argument('--raw', '-r', required=False, help='Path to raw CSV (user-provided)')
    parser.add_argument('--preprocessed-out', '-p', default=r'data/preprocessed_data.csv', help='Output path for preprocessed CSV')
    # CHANGE 3: Update default model name
    parser.add_argument('--model', '-m', default=r'pemfc_model.pkl', help='Path to trained model (pkl)')
    parser.add_argument('--predictions-out', '-o', default=r'data/predictions.csv', help='Output path for predictions CSV')
    parser.add_argument('--plot-out', '-g', default=r'data/predictions_plot.png', help='Output path for the predictions plot (png)')
    parser.add_argument('--show-plot', action='store_true', help='Show the plot interactively after generating it')
    parser.add_argument('--row', '-R', action='append', help='Provide a data row: I, T, Hydrogen, Oxygen, RH anode, Rh Cathode')
    parser.add_argument('--skip-scaling', action='store_true', help='Skip external StandardScaler (Required if model is a Pipeline)')
    parser.add_argument('--save-preprocessed', action='store_true', help='Save preprocessed CSV')
    parser.add_argument('--overwrite-preprocessed', action='store_true', help='Overwrite existing preprocessed file')

    args = parser.parse_args()

    # CLI Row handling
    temp_raw_path = None
    if args.row and not args.raw:
        rows = []
        for r in args.row:
            parts = [p.strip() for p in r.split(',')]
            if len(parts) != len(FEATURES):
                raise ValueError(f"Each --row must contain {len(FEATURES)} values: {', '.join(FEATURES)}")
            try:
                vals = [float(x) for x in parts]
            except Exception:
                raise ValueError(f"Values must be numeric: '{r}'")
            rows.append(vals)
        df_input = pd.DataFrame(rows, columns=FEATURES)
        temp_raw_path = os.path.abspath(os.path.join('data', 'temp_input_rows.csv'))
        os.makedirs(os.path.dirname(temp_raw_path), exist_ok=True)
        df_input.to_csv(temp_raw_path, index=False)

    chosen_raw = args.raw if args.raw else temp_raw_path

    main(chosen_raw, args.preprocessed_out, args.model, args.predictions_out, plot_out=args.plot_out, show_plot=args.show_plot, skip_scaling=args.skip_scaling, save_preprocessed=args.save_preprocessed, overwrite_preprocessed=args.overwrite_preprocessed)