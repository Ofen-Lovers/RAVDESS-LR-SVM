import pandas as pd
import numpy as np

# --- Configuration ---
# Define the filename provided in your request
# NOTE: I'm assuming the file exists in the current directory for this example.
# You would replace this with the actual path if needed.
file_path = 'Model-V5-output/20250930_183242_01_raw_features_and_labels.csv' 

try:
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(file_path)
    print(f"Successfully loaded '{file_path}'")
    print("="*50)

    # --- 1. Basic DataFrame Information ---
    print("\n--- 1. Basic DataFrame Information ---")
    print(f"Shape of the dataset (rows, columns): {df.shape}")
    print("\nData types and non-null counts:")
    # Use info() for a concise summary of the DataFrame
    df.info()
    print("="*50)

    # --- 2. First 5 Rows of the Dataset ---
    print("\n--- 2. First 5 Rows of the Dataset ---")
    print("This shows the structure and sample values for each feature.")
    # To see all columns in the head(), you might need to set an option first
    pd.set_option('display.max_columns', None)
    print(df.head())
    print("="*50)

    # --- 3. Missing Values Check ---
    print("\n--- 3. Missing Values Check ---")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("No missing values found in the dataset. Data is clean.")
    else:
        print("Number of missing values per column:")
        print(missing_values[missing_values > 0])
    print("="*50)

    # --- 4. Descriptive Statistics for Numerical Features ---
    print("\n--- 4. Descriptive Statistics for Numerical Features ---")
    # Select only numerical columns for statistics
    numerical_df = df.select_dtypes(include=np.number)
    # Using .transpose() for better readability with many columns
    print(numerical_df.describe().transpose())
    print("="*50)

    # --- 5. Analysis of Categorical Features ---
    print("\n--- 5. Analysis of Categorical Features ---")
    
    # Analyze the 'emotion' column
    if 'emotion' in df.columns:
        print("\nDistribution of Emotions:")
        print(df['emotion'].value_counts())
        print(f"\nNumber of unique emotions: {df['emotion'].nunique()}")
    else:
        print("\n'emotion' column not found.")

    # Analyze the 'speaker' column
    if 'speaker' in df.columns:
        print("\nDistribution of Samples per Speaker:")
        # Display the top 10 speakers by sample count for brevity
        print(df['speaker'].value_counts().head(10))
        print(f"\nNumber of unique speakers: {df['speaker'].nunique()}")
    else:
        print("\n'speaker' column not found.")
    print("="*50)

    # --- 6. Full List of Feature Columns ---
    print("\n--- 6. Full List of Feature Columns ---")
    # Get the list of all column names
    all_columns = df.columns.tolist()
    print(f"Total number of columns: {len(all_columns)}")
    print("\nColumn Names:")
    # Print each column name for better readability
    for col in all_columns:
        print(col)
    print("="*50)


except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the CSV file is in the same directory as this script, or provide the full path.")
except Exception as e:
    print(f"An error occurred: {e}")