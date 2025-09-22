import os
import warnings
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# --- Configuration ---
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
N_MFCC = 13  # Number of MFCCs to extract

# --- File Path Configuration ---
# Set your data directory path here
DATA_DIR = r"\archive"
OUTPUT_DIR = "output"

# --- Helper Functions ---

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

def extract_features(file_path):
    """
    Extracts a set of acoustic features from a single audio file.
    Returns a dictionary of features or None if an error occurs.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Pitch features
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        
        features = {
            'pitch_mean': np.nanmean(f0) if np.any(~np.isnan(f0)) else 0,
            'pitch_std': np.nanstd(f0) if np.any(~np.isnan(f0)) else 0,
            'energy_mean': np.mean(librosa.feature.rms(y=y)),
            'zcr_mean': np.mean(librosa.feature.zero_crossing_rate(y)),
            'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'chroma_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        }
        
        # Add MFCC means and stds
        for i in range(N_MFCC):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            
        return features
        
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

def _process_file(file_path):
    """Internal helper for parallel processing. Extracts features and label."""
    emotions = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    filename = os.path.basename(file_path)
    parts = filename.split('-')
    
    if len(parts) >= 3:
        emotion_code = parts[2]
        emotion = emotions.get(emotion_code)
        if emotion:
            features = extract_features(file_path)
            if features:
                return features, emotion
    return None, None

def load_and_extract(base_path):
    """
    Loads audio files from the RAVDESS dataset and extracts features in parallel.
    """
    print("Finding all .wav files...")
    all_file_paths = [os.path.join(root, file) 
                      for root, _, files in os.walk(base_path) 
                      for file in files if file.endswith('.wav')]
    
    if not all_file_paths:
        raise FileNotFoundError(f"No .wav files found in {base_path}. Please check the data directory.")

    print(f"Extracting features from {len(all_file_paths)} files using parallel processing...")
    
    # Use joblib for parallel feature extraction
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_process_file)(fp) for fp in tqdm(all_file_paths)
    )
    
    # Filter out any processing errors (None results)
    processed_results = [res for res in results if res[0] is not None]
    if not processed_results:
        raise ValueError("Feature extraction failed for all files. Please check audio files and dependencies.")
        
    features_list, labels_list = zip(*processed_results)
    
    features_df = pd.DataFrame(features_list)
    labels_series = pd.Series(labels_list, name="emotion")
    
    return features_df, labels_series

def preprocess_data(features_df, labels_series, output_dir):
    """
    Preprocesses the data: handles missing values, encodes labels, splits, and scales.
    """
    print("Preprocessing data...")
    
    # Handle missing values using median imputation - more robust to outliers than mean.
    if features_df.isnull().sum().sum() > 0:
        print(f"Found {features_df.isnull().sum().sum()} missing values. Imputing with median.")
        features_df = features_df.fillna(features_df.median())
    
    save_artifact(features_df, "02_features_cleaned.csv", output_dir, "cleaned features")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels_series)
    
    label_mapping = pd.DataFrame({'emotion': label_encoder.classes_, 'encoded_label': range(len(label_encoder.classes_))})
    save_artifact(label_mapping, "02_label_mapping.csv", output_dir, "label encoding map")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder, features_df.columns.tolist()

def train_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test, model_name, label_encoder, output_dir):
    """
    Trains a model with hyperparameter tuning and evaluates its performance.
    """
    print(f"\n=== Training and Evaluating {model_name} ===")
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    print(f"Test Accuracy for {model_name}: {metrics['accuracy']:.4f}")
    
    # Save artifacts
    model_prefix = model_name.lower().replace(' ', '_')
    save_artifact(pd.DataFrame(grid_search.cv_results_), f"03_{model_prefix}_cv_results.csv", output_dir, f"{model_name} CV results")
    save_artifact(pd.DataFrame([metrics]), f"03_{model_prefix}_metrics.csv", output_dir, f"{model_name} metrics")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_artifact(fig, f"03_{model_prefix}_confusion_matrix.png", output_dir)
    
    # Classification Report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    save_artifact(pd.DataFrame(report).transpose(), f"03_{model_prefix}_classification_report.csv", output_dir, f"{model_name} report")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return best_model, metrics

def analyze_feature_importance(model, X_test, y_test, feature_names, model_name, output_dir):
    """
    Analyzes and plots feature importance for a given model.
    Handles both linear and non-linear models.
    """
    print(f"\nAnalyzing feature importance for {model_name}...")
    model_prefix = model_name.lower().replace(' ', '_')
    
    # For linear models, coefficients are readily available and fast to compute.
    if isinstance(model, SVC) and model.kernel == 'linear':##########################################################
        importance = np.abs(model.coef_).mean(axis=0)
        importance_type = "Coefficient-based"
    # For non-linear models or as a general robust method, use permutation importance.
    else:
        print("Using permutation importance (this may take a moment)...")
        importance_type = "Permutation-based"
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        importance = result.importances_mean

    feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feat_imp_df = feat_imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    save_artifact(feat_imp_df, f"04_{model_prefix}_feature_importance.csv", output_dir, f"{model_name} feature importance")
    
    # Plot top 20 features
    fig = plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feat_imp_df.head(20))
    plt.title(f'Top 20 Feature Importance ({importance_type}) - {model_name}')
    plt.tight_layout()
    save_artifact(fig, f"04_{model_prefix}_feature_importance.png", output_dir)
    
    return feat_imp_df

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
        features_dict = extract_features(audio_file_path)
        if features_dict is None:
            return {"error": "Could not extract features from the audio file."}
        
        # Ensure feature vector is in the same order as during training
        feature_vector = [features_dict.get(name, 0) for name in self.feature_names]
        
        # Scale and predict
        scaled_features = self.scaler.transform([feature_vector])
        prediction_encoded = self.model.predict(scaled_features)
        prediction_proba = self.model.predict_proba(scaled_features)
        
        # Format output
        predicted_emotion = self.label_encoder.inverse_transform(prediction_encoded)[0]
        confidence = {emotion: prob for emotion, prob in zip(self.label_encoder.classes_, prediction_proba[0])}
        
        return {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence
        }

# --- Main Execution Block ---

def main():
    """Main function to run the entire pipeline."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")
    print(f"Data directory: {DATA_DIR}")

    try:
        # 1. Data Loading and Feature Extraction
        print("STEP 1: Loading data and extracting features")
        features_df, labels_series = load_and_extract(DATA_DIR)
        save_artifact(features_df.join(labels_series), "01_raw_features_and_labels.csv", OUTPUT_DIR, "raw extracted features")

        # 2. Preprocessing
        print("STEP 2: Preprocessing data")
        X_train, X_test, y_train, y_test, scaler, le, feature_names = preprocess_data(features_df, labels_series, OUTPUT_DIR)

        # 3. Model Training and Evaluation
        print("STEP 3: Training and evaluating models")
        
        # SVM
        svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto'], 'probability': [True], 'random_state': [RANDOM_STATE]}
        svm_model, svm_metrics = train_and_evaluate_model(SVC(), svm_param_grid, X_train, y_train, X_test, y_test, "SVM", le, OUTPUT_DIR)
        save_artifact(svm_model, "03_svm_model.pkl", OUTPUT_DIR, "Trained SVM model")

        # Logistic Regression
        lr_param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs'], 'max_iter': [1000], 'random_state': [RANDOM_STATE]}
        lr_model, lr_metrics = train_and_evaluate_model(LogisticRegression(), lr_param_grid, X_train, y_train, X_test, y_test, "Logistic Regression", le, OUTPUT_DIR)
        save_artifact(lr_model, "03_logistic_regression_model.pkl", OUTPUT_DIR, "Trained LR model")

        # 4. Analysis and Comparison
        print("STEP 4: Analyzing feature importance and comparing models")
        svm_feat_imp = analyze_feature_importance(svm_model, X_test, y_test, feature_names, "SVM", OUTPUT_DIR)
        lr_feat_imp = analyze_feature_importance(lr_model, X_test, y_test, feature_names, "Logistic Regression", OUTPUT_DIR)

        comparison_df = pd.DataFrame({'SVM': svm_metrics, 'Logistic Regression': lr_metrics}).T
        print("\n=== Model Performance Comparison ===")
        print(comparison_df)
        save_artifact(comparison_df, "04_model_comparison.csv", OUTPUT_DIR, "Model comparison")

        # 5. Finalize and Save Best Model Pipeline
        print("STEP 5: Finalizing and saving best model")
        best_model_name = comparison_df['accuracy'].idxmax()
        best_model = svm_model if best_model_name == "SVM" else lr_model
        print(f"\nBest performing model is: {best_model_name} with accuracy {comparison_df.loc[best_model_name, 'accuracy']:.4f}")

        predictor = EmotionPredictor(
            model=best_model,
            scaler=scaler,
            label_encoder=le,
            feature_names=feature_names
        )
        save_artifact(predictor, "05_emotion_predictor.pkl", OUTPUT_DIR, "Final predictor object")

        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("To use the final model, load '05_emotion_predictor.pkl' and call its .predict() method with a .wav file path.")
        print(f"All output files are saved in: {OUTPUT_DIR}")

    except Exception as e:
        print(f"\nError occurred during pipeline execution: {e}")
        print("Please check the data directory path and ensure all dependencies are installed.")
        raise

if __name__ == "__main__":
    main()