# main.py

import os
import warnings
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Importing from our custom modules ---
from feature_extraction import load_and_extract
from preprocessing import preprocess_data
from svm_model_training import train_svm
from logistic_regression_model_training import train_logistic_regression
from evaluation import analyze_feature_importance

# --- Configuration ---
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
N_MFCC = 13  # Number of MFCCs to extract

# --- File Path Configuration ---
DATA_DIR = "archive"  # Use a relative path
OUTPUT_DIR = "output-using-deltacoefficients"

# --- Utility Functions ---
def save_artifact(data, filename, output_dir, description=""):
    """Helper function to save data (DataFrame, model, etc.) with a timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{timestamp}_{filename}")
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
        print(f"Saved {description} to: {filepath}")
    elif isinstance(data, plt.Figure):
        data.savefig(filepath)
        plt.close(data)
        print(f"Saved plot to: {filepath}")
    else:
        joblib.dump(data, filepath)
        print(f"Saved object to: {filepath}")
    return filepath

# --- Predictor Class for Deployment ---
class EmotionPredictor:
    """A class to encapsulate the trained pipeline for easy prediction."""
    def __init__(self, model, scaler, label_encoder, feature_names):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.feature_names = feature_names

    def predict(self, audio_file_path):
        """Predicts the emotion from a single audio file."""
        from feature_extraction import extract_features
        
        features_dict = extract_features(audio_file_path, n_mfcc=N_MFCC)
        if features_dict is None:
            return {"error": "Could not extract features from the audio file."}
        
        feature_vector = [features_dict.get(name, 0) for name in self.feature_names]
        scaled_features = self.scaler.transform([feature_vector])
        prediction_encoded = self.model.predict(scaled_features)
        prediction_proba = self.model.predict_proba(scaled_features)
        
        predicted_emotion = self.label_encoder.inverse_transform(prediction_encoded)[0]
        confidence = {emotion: prob for emotion, prob in zip(self.label_encoder.classes_, prediction_proba[0])}
        
        return {'predicted_emotion': predicted_emotion, 'confidence': confidence}

# --- Main Execution Block ---
def main():
    """Main function to run the entire pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")
    print(f"Data directory: {DATA_DIR}")

    try:
        # 1. Data Loading and Feature Extraction
        print("STEP 1: Loading data and extracting features")
        features_df, labels_series = load_and_extract(DATA_DIR, n_mfcc=N_MFCC)
        save_artifact(features_df.join(labels_series), "01_raw_features_and_labels.csv", OUTPUT_DIR, "raw extracted features")

        # 2. Preprocessing
        print("\nSTEP 2: Preprocessing data")
        processed_data = preprocess_data(features_df, labels_series, OUTPUT_DIR, RANDOM_STATE, save_artifact)
        X_train, X_test, y_train, y_test = processed_data['X_train'], processed_data['X_test'], processed_data['y_train'], processed_data['y_test']
        scaler, le, feature_names = processed_data['scaler'], processed_data['label_encoder'], processed_data['feature_names']

        # 3. Model Training
        print("\nSTEP 3: Training and evaluating models")
        svm_model, svm_metrics = train_svm(X_train, y_train, X_test, y_test, le, OUTPUT_DIR, RANDOM_STATE, save_artifact)
        lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test, le, OUTPUT_DIR, RANDOM_STATE, save_artifact)

        # 4. Analysis and Comparison
        print("\nSTEP 4: Analyzing feature importance and comparing models")
        analyze_feature_importance(svm_model, X_test, y_test, feature_names, "SVM", OUTPUT_DIR, RANDOM_STATE, save_artifact)
        analyze_feature_importance(lr_model, X_test, y_test, feature_names, "Logistic Regression", OUTPUT_DIR, RANDOM_STATE, save_artifact)

        comparison_df = pd.DataFrame({'SVM': svm_metrics, 'Logistic Regression': lr_metrics}).T
        print("\n=== Model Performance Comparison ===")
        print(comparison_df)
        save_artifact(comparison_df, "04_model_comparison.csv", OUTPUT_DIR, "Model comparison")

        # 5. Finalize and Save Best Model Pipeline
        print("\nSTEP 5: Finalizing and saving best model")
        best_model_name = comparison_df['accuracy'].idxmax()
        best_model = svm_model if best_model_name == "SVM" else lr_model
        print(f"\nBest performing model is: {best_model_name} with accuracy {comparison_df.loc[best_model_name, 'accuracy']:.4f}")

        predictor = EmotionPredictor(model=best_model, scaler=scaler, label_encoder=le, feature_names=feature_names)
        save_artifact(predictor, "05_emotion_predictor.pkl", OUTPUT_DIR, "Final predictor object")

        print("\nPIPELINE COMPLETED SUCCESSFULLY!")

    except Exception as e:
        print(f"\nError occurred during pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()