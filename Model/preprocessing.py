# preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(features_df, labels_series, output_dir, random_state, save_artifact_func):
    """Preprocesses the data: encodes labels, splits data, and scales features."""
    print("Preprocessing data...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels_series)
    label_mapping = pd.DataFrame({'emotion': label_encoder.classes_, 'encoded_label': range(len(label_encoder.classes_))})
    save_artifact_func(label_mapping, "02_label_mapping.csv", output_dir, "label encoding map")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded
    )
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "label_encoder": label_encoder,
        "feature_names": features_df.columns.tolist()
    }