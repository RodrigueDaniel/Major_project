import os
import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
# New Input Order: Current, Temp, H2 Flow, O2 Flow, RH Anode, RH Cathode
FEATURES = ['I', 'T', 'Hydrogen', 'Oxygen', 'RH anode', 'Rh Cathode']
DEFAULT_MODEL_PATH = r'D:\Coding\Major-Project\new_\new\pemfc_model.pkl'

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Did you run train_model.py first?")
    return joblib.load(model_path)

def main(args):
    # 1. Parse Input Data
    df_input = None
    
    if args.row:
        # User provided rows via command line
        rows = []
        for r in args.row:
            parts = [p.strip() for p in r.split(',')]
            if len(parts) != len(FEATURES):
                raise ValueError(f"Input Error: Expected {len(FEATURES)} values. Got {len(parts)}.\nExpected order: I, T, H2, O2, RHa, RHc")
            rows.append([float(x) for x in parts])
        df_input = pd.DataFrame(rows, columns=FEATURES)
    
    elif args.raw:
        # User provided a CSV file
        if not os.path.exists(args.raw):
            raise FileNotFoundError(f"Input file not found: {args.raw}")
        df_input = pd.read_csv(args.raw)
        # Ensure only the necessary columns are used
        try:
            df_input = df_input[FEATURES]
        except KeyError as e:
            print(f"Error: Your input CSV is missing columns: {e}")
            return
    
    else:
        print("Error: No input provided. Use --row 'val1,val2...' or --raw 'file.csv'")
        return

    # 2. Load Model
    try:
        pipeline = load_model(args.model)
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        return

    # 3. Predict
    print(f"Running prediction on {len(df_input)} samples...")
    predictions = pipeline.predict(df_input)

    # 4. Display Results
    print("\n" + "="*40)
    print(f" PEMFC VOLTAGE PREDICTION")
    print("="*40)
    
    df_output = df_input.copy()
    df_output['Predicted_Voltage_V'] = predictions
    
    # Print first few results to console
    for i, row in df_output.head(5).iterrows():
        print(f"Sample {i+1}:")
        print(f"  Current:   {row['I']:.2f} A")
        print(f"  Temp:      {row['T']:.2f} C")
        print(f"  Flows:     H2={row['Hydrogen']:.4f}, O2={row['Oxygen']:.4f}")
        print(f"  PREDICTED: {row['Predicted_Voltage_V']:.4f} V")
        print("-" * 20)

    # 5. Save Results
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        df_output.to_csv(args.out, index=False)
        print(f"\nFull results saved to: {args.out}")

    # 6. Plotting (Optional)
    if not args.no_plot and len(df_output) > 1:
        plt.figure(figsize=(8, 5))
        # Sort by Current to draw a nice line
        df_sorted = df_output.sort_values('I')
        plt.plot(df_sorted['I'], df_sorted['Predicted_Voltage_V'], color='red', label='Prediction', linewidth=2)
        plt.scatter(df_output['I'], df_output['Predicted_Voltage_V'], color='blue', s=20, alpha=0.6)
        
        plt.xlabel('Current (A)')
        plt.ylabel('Voltage (V)')
        plt.title('Predicted Polarization Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save Plot
        plot_path = args.out.replace('.csv', '.png') if args.out else 'prediction_plot.png'
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        
        if args.show_plot:
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict PEM Fuel Cell Voltage')
    parser.add_argument('--row', '-R', action='append', help='Input row: "I, T, H2, O2, RHa, RHc"')
    parser.add_argument('--raw', '-r', help='Path to input CSV file containing multiple rows')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL_PATH, help='Path to .pkl model file')
    parser.add_argument('--out', '-o', default='predictions.csv', help='Path to save output CSV')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--show-plot', action='store_true', help='Show plot window')

    args = parser.parse_args()
    main(args)