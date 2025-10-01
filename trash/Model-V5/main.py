import os
import warnings
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import LabelEncoder

# Configuration
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
N_MFCC = 13

DATA_DIR = "archive-16khz-v2"
OUTPUT_DIR = "Model-V6-output-TEST"

def save_artifact(data, filename, output_dir, description=""):
    """Helper function to save data with a timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{timestamp}_{filename}")
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
        print(f"Saved {description} to: {filepath}")
    elif isinstance(data, plt.Figure):
        data.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(data)
        print(f"Saved plot to: {filepath}")
    else:
        joblib.dump(data, filepath)
        print(f"Saved object to: {filepath}")
    return filepath

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")
    print(f"Data directory: {DATA_DIR}")

    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ ERROR: Data directory '{DATA_DIR}' not found!")
        print("Please check the path and try again.")
        return

    try:
        # 1. Data Loading and Feature Extraction (TEST MODE)
        print("STEP 1: Loading data and extracting features (TEST MODE)")
        
        # Import here to get the updated function
        from feature_extraction import load_and_extract_safe
        
        # Test with only 10 files first
        features_df, labels_series, groups_series = load_and_extract_safe(
            DATA_DIR, 
            n_mfcc=N_MFCC, 
            max_files=10  # Start with just 10 files
        )
        
        # Save what we have
        raw_data = features_df.copy()
        raw_data['emotion'] = labels_series.values
        raw_data['speaker'] = groups_series.values
        save_artifact(raw_data, "01_test_features.csv", OUTPUT_DIR, "test features")
        
        print("✅ Feature extraction completed successfully!")
        print(f"Dataset shape: {features_df.shape}")
        
        # If this works, you can remove max_files limit for full processing
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()