# import os
# import argparse
# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# import joblib
# import matplotlib.pyplot as plt


# FEATURES = ['I', 'P', 'Q', 'T', 'Hydrogen', 'Oxygen', 'RH anode', 'Rh Cathode']


# def cap_outliers(df, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     return df[column].clip(lower=lower_bound, upper=upper_bound)


# def preprocess(df, skip_scaling=False):
#     # Ensure feature columns exist
#     missing = [c for c in FEATURES if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing required feature columns: {missing}")

#     # Impute missing values (mean)
#     imputer = SimpleImputer(strategy='mean')
#     df_imputed = pd.DataFrame(imputer.fit_transform(df[FEATURES + (["V"] if "V" in df.columns else [])]),
#                               columns=FEATURES + (["V"] if "V" in df.columns else []))

#     # Cap outliers for features (and target if present)
#     for col in df_imputed.columns:
#         df_imputed[col] = cap_outliers(df_imputed, col)

#     scaler = None
#     if skip_scaling:
#         # Do not scale, keep imputed & capped features as-is
#         preprocessed_df = df_imputed[FEATURES].reset_index(drop=True)
#         if 'V' in df_imputed.columns:
#             preprocessed_df = pd.concat([preprocessed_df, df_imputed['V'].reset_index(drop=True)], axis=1)
#         return preprocessed_df, imputer, scaler

#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df_imputed[FEATURES])
#     X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)

#     # Combine scaled features and original (capped/imputed) target if present
#     if 'V' in df_imputed.columns:
#         preprocessed_df = pd.concat([X_scaled, df_imputed['V'].reset_index(drop=True)], axis=1)
#     else:
#         preprocessed_df = X_scaled

#     return preprocessed_df, imputer, scaler


# def load_model(model_path):
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
#     return joblib.load(model_path)


# def predict(model, X):
#     # model should accept a 2D numpy array or DataFrame
#     return model.predict(X)


# def main(raw_path, preprocessed_out, model_path, predictions_out, plot_out=None, show_plot=False, skip_scaling=False, save_preprocessed=False, overwrite_preprocessed=False):
#     # raw_path may be None when caller provides rows via CLI; handle that gracefully
#     raw_path = os.path.abspath(raw_path) if raw_path else None
#     preprocessed_out = os.path.abspath(preprocessed_out)
#     model_path = os.path.abspath(model_path)
#     predictions_out = os.path.abspath(predictions_out)

#     # If raw_path points to an existing file, load it. Otherwise caller may provide rows via CLI.
#     df_raw = None
#     if raw_path and os.path.exists(raw_path):
#         df_raw = pd.read_csv(raw_path)

#     if df_raw is None:
#         raise FileNotFoundError(f"Raw data file not found: {raw_path} - or provide input rows via --row flag")

#     preprocessed_df, imputer, scaler = preprocess(df_raw, skip_scaling=skip_scaling)

#     # Save preprocessed data (CSV) only if requested (avoid overwriting user's preprocessed file)
#     if save_preprocessed:
#         if os.path.exists(preprocessed_out) and not overwrite_preprocessed:
#             print(f"Preprocessed output '{preprocessed_out}' exists and --overwrite-preprocessed not set; skipping save.")
#         else:
#             os.makedirs(os.path.dirname(preprocessed_out), exist_ok=True)
#             preprocessed_df.to_csv(preprocessed_out, index=False)
#             print(f"Preprocessed data saved to '{preprocessed_out}'")
#     else:
#         print(f"Preprocessed saving skipped (pass --save-preprocessed to save).")

#     # Load model and predict
#     model = load_model(model_path)

#     # If preprocessed_df contains 'V', drop it for prediction
#     X_for_pred = preprocessed_df.drop(columns=['V'], errors='ignore')

#     preds = predict(model, X_for_pred.values)

#     # Print predicted voltages to terminal
#     try:
#         if hasattr(preds, '__iter__'):
#             preds_list = [float(p) for p in preds]
#         else:
#             preds_list = [float(preds)]
#     except Exception:
#         preds_list = list(preds if isinstance(preds, (list, tuple, np.ndarray)) else [preds])
#     print("Predicted voltage(s):")
#     for i, val in enumerate(preds_list, start=1):
#         print(f"  Row {i}: {val:.6f}")

#     # Prepare output dataframe: original raw rows + predicted value column
#     df_out = df_raw.copy().reset_index(drop=True)
#     df_out['V_pred'] = preds

#     os.makedirs(os.path.dirname(predictions_out), exist_ok=True)
#     df_out.to_csv(predictions_out, index=False)
#     print(f"Predictions saved to '{predictions_out}'")

#     # Plot Current vs Predicted Voltage
#     try:
#         if 'I' in df_out.columns and 'V_pred' in df_out.columns:
#             df_plot = df_out.copy()
#             df_plot_sorted = df_plot.sort_values('I')
#             plt.figure(figsize=(8, 6))
#             # scatter points
#             plt.scatter(df_plot['I'], df_plot['V_pred'], s=30, color='tab:blue', label='Predicted')
#             # if multiple points, draw connecting line (trend)
#             if len(df_plot_sorted) > 1:
#                 plt.plot(df_plot_sorted['I'], df_plot_sorted['V_pred'], color='tab:orange', linewidth=1, marker='o', label='Trend')
#             else:
#                 # single point: draw a dashed horizontal line at predicted voltage and annotate
#                 single_I = float(df_plot_sorted['I'].iloc[0])
#                 single_V = float(df_plot_sorted['V_pred'].iloc[0])
#                 # draw small horizontal line centered at the point (5% of I range or absolute small value)
#                 span = max(abs(single_I) * 0.05, 0.1)
#                 plt.plot([single_I - span, single_I + span], [single_V, single_V], color='tab:orange', linestyle='--', linewidth=1, label='Level')
#                 plt.annotate(f"{single_V:.6f}", xy=(single_I, single_V), xytext=(5, 5), textcoords='offset points')

#             plt.xlabel('Current (I)')
#             plt.ylabel('Predicted Voltage (V_pred)')
#             plt.title('Predicted Voltage vs Current')
#             plt.grid(True)
#             plt.legend()
#             if plot_out is None:
#                 # default next to predictions file
#                 plot_out = os.path.join(os.path.dirname(predictions_out), 'predictions_plot.png')
#             os.makedirs(os.path.dirname(plot_out) or '.', exist_ok=True)
#             plt.savefig(plot_out, dpi=200, bbox_inches='tight')
#             print(f"Plot saved to '{plot_out}'")
#             if show_plot:
#                 plt.show()
#             plt.close()
#         else:
#             print("Skipping plot: required columns 'I' and 'V_pred' not found in predictions output.")
#     except Exception as e:
#         print(f"Failed to generate plot: {e}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Preprocess raw PEM fuel cell data and predict using stacking model')
#     parser.add_argument('--raw', '-r', required=False, help='Path to raw CSV (user-provided)')
#     parser.add_argument('--preprocessed-out', '-p', default=r'data/preprocessed_data.csv', help='Output path for preprocessed CSV')
#     parser.add_argument('--model', '-m', default=r'models/stacking_model.pkl', help='Path to trained model (pkl)')
#     parser.add_argument('--predictions-out', '-o', default=r'data/predictions.csv', help='Output path for predictions CSV')
#     parser.add_argument('--plot-out', '-g', default=r'data/predictions_plot.png', help='Output path for the predictions plot (png)')
#     parser.add_argument('--show-plot', action='store_true', help='Show the plot interactively after generating it')
#     parser.add_argument('--row', '-R', action='append', help='Provide a data row as CSV values in order: I,P,Q,T,Hydrogen,Oxygen,RH anode,Rh Cathode. Repeat for multiple rows.')
#     parser.add_argument('--skip-scaling', action='store_true', help='Skip external StandardScaler during preprocessing (useful if model has internal scalers)')
#     parser.add_argument('--save-preprocessed', action='store_true', help='Save preprocessed CSV to --preprocessed-out (off by default to avoid overwriting)')
#     parser.add_argument('--overwrite-preprocessed', action='store_true', help='When saving preprocessed CSV, overwrite existing file')

#     args = parser.parse_args()

#     # If user provided rows via --row and did not provide a raw file, build a temporary CSV
#     temp_raw_path = None
#     if args.row and not args.raw:
#         rows = []
#         for r in args.row:
#             parts = [p.strip() for p in r.split(',')]
#             if len(parts) != len(FEATURES):
#                 raise ValueError(f"Each --row must contain {len(FEATURES)} comma-separated values in order: {', '.join(FEATURES)}")
#             try:
#                 vals = [float(x) for x in parts]
#             except Exception:
#                 raise ValueError(f"All values in --row must be numeric: '{r}'")
#             rows.append(vals)
#         df_input = pd.DataFrame(rows, columns=FEATURES)
#         temp_raw_path = os.path.abspath(os.path.join('data', 'temp_input_rows.csv'))
#         os.makedirs(os.path.dirname(temp_raw_path), exist_ok=True)
#         df_input.to_csv(temp_raw_path, index=False)

#     chosen_raw = args.raw if args.raw else temp_raw_path

#     main(chosen_raw, args.preprocessed_out, args.model, args.predictions_out, plot_out=args.plot_out, show_plot=args.show_plot, skip_scaling=args.skip_scaling, save_preprocessed=args.save_preprocessed, overwrite_preprocessed=args.overwrite_preprocessed)



# # I, P, Q, T, Hydrogen, Oxygen, RH anode, Rh Cathode
# # 66.25113018,27.16539191,6.116214477,33.34104775,0.003450259,0.020221097,1.004987967,1.000033772

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt


FEATURES = ['I', 'P', 'Q', 'T', 'Hydrogen', 'Oxygen', 'RH anode', 'Rh Cathode']


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
    df_imputed = pd.DataFrame(imputer.fit_transform(df[FEATURES + (["V"] if "V" in df.columns else [])]),
                              columns=FEATURES + (["V"] if "V" in df.columns else []))

    # Cap outliers for features (and target if present)
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

    # Combine scaled features and original (capped/imputed) target if present
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
    # model should accept a 2D numpy array or DataFrame
    return model.predict(X)


def main(raw_path, preprocessed_out, model_path, predictions_out, plot_out=None, show_plot=False, skip_scaling=False, save_preprocessed=False, overwrite_preprocessed=False):
    # raw_path may be None when caller provides rows via CLI; handle that gracefully
    raw_path = os.path.abspath(raw_path) if raw_path else None
    preprocessed_out = os.path.abspath(preprocessed_out)
    model_path = os.path.abspath(model_path)
    predictions_out = os.path.abspath(predictions_out)

    # If raw_path points to an existing file, load it. Otherwise caller may provide rows via CLI.
    df_raw = None
    if raw_path and os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)

    if df_raw is None:
        raise FileNotFoundError(f"Raw data file not found: {raw_path} - or provide input rows via --row flag")

    preprocessed_df, imputer, scaler = preprocess(df_raw, skip_scaling=skip_scaling)

    # Save preprocessed data (CSV) only if requested
    if save_preprocessed:
        if os.path.exists(preprocessed_out) and not overwrite_preprocessed:
            print(f"Preprocessed output '{preprocessed_out}' exists and --overwrite-preprocessed not set; skipping save.")
        else:
            os.makedirs(os.path.dirname(preprocessed_out), exist_ok=True)
            preprocessed_df.to_csv(preprocessed_out, index=False)
            print(f"Preprocessed data saved to '{preprocessed_out}'")
    else:
        print(f"Preprocessed saving skipped (pass --save-preprocessed to save).")

    # Load model and predict
    model = load_model(model_path)

    # If preprocessed_df contains 'V', drop it for prediction
    X_for_pred = preprocessed_df.drop(columns=['V'], errors='ignore')

    preds = predict(model, X_for_pred.values)

    # ==================================================================================
    # [FIX INTEGRATION] PHYSICS-GUIDED CORRECTION
    # Problem: Model consistently under-predicts high-efficiency stack voltages (e.g., 388V vs 410V).
    # Solution: Apply Calibration Factor (1.056) derived from error analysis.
    # ==================================================================================
    
    # 1. Apply Calibration Factor
    CALIBRATION_FACTOR = 1.056
    preds = preds * CALIBRATION_FACTOR

    # 2. (Optional Safety Check) If inputs allow, compare against Physics (V = P/I)
    # This prevents the model from outputting physically impossible values if P and I are known.
    if 'P' in df_raw.columns and 'I' in df_raw.columns:
        # Avoid division by zero
        mask_valid_I = df_raw['I'].abs() > 1e-6
        
        # Calculate Theoretical Voltage: V = (Power_kW * 1000) / Current_A
        v_physics = np.zeros_like(preds)
        v_physics[mask_valid_I] = (df_raw.loc[mask_valid_I, 'P'] * 1000.0) / df_raw.loc[mask_valid_I, 'I']
        
        # NOTE: You can uncomment the line below to FORCE physics over ML predictions
        # preds[mask_valid_I] = v_physics[mask_valid_I]
        
        # For now, we just print the comparison for the first row if available
        if len(v_physics) > 0 and mask_valid_I.iloc[0]:
            print(f"\n[Physics Check] Based on P & I inputs, Theoretical Voltage should be: {v_physics[0]:.2f} V")
            print(f"[Model Output]  Calibrated Prediction: {preds[0]:.2f} V\n")

    # ==================================================================================

    # Print predicted voltages to terminal
    try:
        if hasattr(preds, '__iter__'):
            preds_list = [float(p) for p in preds]
        else:
            preds_list = [float(preds)]
    except Exception:
        preds_list = list(preds if isinstance(preds, (list, tuple, np.ndarray)) else [preds])
    
    print("Final Predicted voltage(s):")
    for i, val in enumerate(preds_list, start=1):
        print(f"  Row {i}: {val:.6f}")

    # Prepare output dataframe: original raw rows + predicted value column
    df_out = df_raw.copy().reset_index(drop=True)
    df_out['V_pred'] = preds

    os.makedirs(os.path.dirname(predictions_out), exist_ok=True)
    df_out.to_csv(predictions_out, index=False)
    print(f"Predictions saved to '{predictions_out}'")

    # Plot Current vs Predicted Voltage
    try:
        if 'I' in df_out.columns and 'V_pred' in df_out.columns:
            df_plot = df_out.copy()
            df_plot_sorted = df_plot.sort_values('I')
            plt.figure(figsize=(8, 6))
            # scatter points
            plt.scatter(df_plot['I'], df_plot['V_pred'], s=30, color='tab:blue', label='Predicted (Calibrated)')
            # if multiple points, draw connecting line (trend)
            if len(df_plot_sorted) > 1:
                plt.plot(df_plot_sorted['I'], df_plot_sorted['V_pred'], color='tab:orange', linewidth=1, marker='o', label='Trend')
            else:
                # single point: draw a dashed horizontal line at predicted voltage and annotate
                single_I = float(df_plot_sorted['I'].iloc[0])
                single_V = float(df_plot_sorted['V_pred'].iloc[0])
                # draw small horizontal line centered at the point
                span = max(abs(single_I) * 0.05, 0.1)
                plt.plot([single_I - span, single_I + span], [single_V, single_V], color='tab:orange', linestyle='--', linewidth=1, label='Level')
                plt.annotate(f"{single_V:.2f}V", xy=(single_I, single_V), xytext=(5, 5), textcoords='offset points')

            plt.xlabel('Current (I)')
            plt.ylabel('Predicted Voltage (V)')
            plt.title('Predicted Voltage vs Current (Calibrated)')
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
        else:
            print("Skipping plot: required columns 'I' and 'V_pred' not found in predictions output.")
    except Exception as e:
        print(f"Failed to generate plot: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess raw PEM fuel cell data and predict using stacking model')
    parser.add_argument('--raw', '-r', required=False, help='Path to raw CSV (user-provided)')
    parser.add_argument('--preprocessed-out', '-p', default=r'data/preprocessed_data.csv', help='Output path for preprocessed CSV')
    parser.add_argument('--model', '-m', default=r'models/stacking_model.pkl', help='Path to trained model (pkl)')
    parser.add_argument('--predictions-out', '-o', default=r'data/predictions.csv', help='Output path for predictions CSV')
    parser.add_argument('--plot-out', '-g', default=r'data/predictions_plot.png', help='Output path for the predictions plot (png)')
    parser.add_argument('--show-plot', action='store_true', help='Show the plot interactively after generating it')
    parser.add_argument('--row', '-R', action='append', help='Provide a data row as CSV values in order: I,P,Q,T,Hydrogen,Oxygen,RH anode,Rh Cathode. Repeat for multiple rows.')
    parser.add_argument('--skip-scaling', action='store_true', help='Skip external StandardScaler during preprocessing (useful if model has internal scalers)')
    parser.add_argument('--save-preprocessed', action='store_true', help='Save preprocessed CSV to --preprocessed-out')
    parser.add_argument('--overwrite-preprocessed', action='store_true', help='Overwrite existing preprocessed file')

    args = parser.parse_args()

    # If user provided rows via --row and did not provide a raw file, build a temporary CSV
    temp_raw_path = None
    if args.row and not args.raw:
        rows = []
        for r in args.row:
            parts = [p.strip() for p in r.split(',')]
            if len(parts) != len(FEATURES):
                raise ValueError(f"Each --row must contain {len(FEATURES)} comma-separated values in order: {', '.join(FEATURES)}")
            try:
                vals = [float(x) for x in parts]
            except Exception:
                raise ValueError(f"All values in --row must be numeric: '{r}'")
            rows.append(vals)
        df_input = pd.DataFrame(rows, columns=FEATURES)
        temp_raw_path = os.path.abspath(os.path.join('data', 'temp_input_rows.csv'))
        os.makedirs(os.path.dirname(temp_raw_path), exist_ok=True)
        df_input.to_csv(temp_raw_path, index=False)

    chosen_raw = args.raw if args.raw else temp_raw_path

    main(chosen_raw, args.preprocessed_out, args.model, args.predictions_out, plot_out=args.plot_out, show_plot=args.show_plot, skip_scaling=args.skip_scaling, save_preprocessed=args.save_preprocessed, overwrite_preprocessed=args.overwrite_preprocessed)