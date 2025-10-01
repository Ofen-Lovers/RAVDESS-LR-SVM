# SVM2.py

# %% Imports
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast # <--- ADDED IMPORT

# %% Feature Extraction
# NOTE: This script assumes audio files have been pre-converted to 16kHz.
dataset_path = "archive-16khz-v2" # Path to your 16kHz audio files
csv_filename = "ravdess_full_features.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The directory '{dataset_path}' was not found. Please ensure your 16kHz audio files are in this folder.")

if os.path.exists(csv_filename):
    print(f"Loading features from '{csv_filename}'")
    df = pd.read_csv(csv_filename)
else:
    print("Extracting a comprehensive set of features from dataset...")
    features_list = []
    file_list = [os.path.join(root, file) for root, _, files in os.walk(dataset_path) for file in files if file.endswith(".wav")]
    
    for file_path in tqdm(file_list, desc="Extracting Features"):
        y, sr = librosa.load(file_path, sr=16000)
        
        # Temporal features
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # MFCC and its derivatives
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        
        # Pitch
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Aggregate features
        features = [
            np.mean(zcr), np.mean(rms), duration, tempo,
            np.mean(spectral_centroid), np.mean(spectral_bandwidth),
            np.mean(spectral_contrast), np.mean(spectral_flatness),
            pitch_mean
        ]
        features.extend([np.mean(mfcc[i]) for i in range(13)])
        features.extend([np.mean(delta_mfcc[i]) for i in range(13)])
        features.extend([np.mean(delta2_mfcc[i]) for i in range(13)])
        features.extend([np.mean(chroma[i]) for i in range(chroma.shape[0])])
        features.extend([np.mean(tonnetz[i]) for i in range(tonnetz.shape[0])])
        
        # Emotion label from filename
        file_name = os.path.basename(file_path)
        emotion = int(file_name.split("-")[2])
        features_list.append(features + [emotion])

    # Define column names for the DataFrame
    columns = [
        "zcr_mean", "rms_mean", "duration_s", "tempo_bpm",
        "spectral_centroid_mean", "spectral_bandwidth_mean",
        "spectral_contrast_mean", "spectral_flatness_mean",
        "pitch_mean"
    ] + [f"mfcc{i+1}" for i in range(13)] \
      + [f"delta_mfcc{i+1}" for i in range(13)] \
      + [f"delta2_mfcc{i+1}" for i in range(13)] \
      + [f"chroma{i+1}" for i in range(12)] \
      + [f"tonnetz{i+1}" for i in range(6)] \
      + ["emotion"]

    df = pd.DataFrame(features_list, columns=columns)
    df.to_csv(csv_filename, index=False)
    print(f"Full features saved to '{csv_filename}'")

# ##################################################################
# %% ADDED SECTION: Data Cleaning
# This step is crucial to fix the ValueError by converting stringified lists to floats.
def clean_tempo(val):
    if isinstance(val, str):
        try:
            # Safely evaluate string representation of list/tuple
            v = ast.literal_eval(val)
            if isinstance(v, (list, tuple, np.ndarray)):
                return float(v[0]) if len(v) > 0 else np.nan
            return float(v)
        except (ValueError, SyntaxError):
            return np.nan
    return val

# Apply the cleaning function to the problematic column
df['tempo_bpm'] = df['tempo_bpm'].apply(clean_tempo)
# Ensure the column is numeric, filling any errors with 0
df['tempo_bpm'] = pd.to_numeric(df['tempo_bpm'], errors='coerce').fillna(0)
print("Cleaned 'tempo_bpm' column.")
# ##################################################################


# %% Data Preparation
# Split features and labels
X = df.drop("emotion", axis=1)
y = df["emotion"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% Model Training
print("\nTraining Linear SVM...")
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)
print("Linear SVM training complete.")

print("\nTraining RBF SVM...")
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)
print("RBF SVM training complete.")

# %% Model Evaluation
def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1-score": f1_score(y_true, y_pred, average='weighted')
    }

# Compare models
results = pd.DataFrame({
    "Linear SVM": evaluate_model(y_test, y_pred_linear),
    "RBF SVM": evaluate_model(y_test, y_pred_rbf)
}).T

print("\n--- Comparative Evaluation Table ---")
print(results)

# %% Confusion Matrix Visualization
print("\nDisplaying Confusion Matrix for the RBF SVM model...")
cm = confusion_matrix(y_test, y_pred_rbf)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (RBF SVM)")
plt.show()

# %% Classification Report and Analysis
print("\n--- Detailed Classification Report (RBF SVM) ---")
print(classification_report(y_test, y_pred_rbf, zero_division=0))

print("\n--- Short Analysis ---")
print("Usually in RAVDESS, emotions like 'calm' and 'neutral' are harder to classify,")
print("as well as 'fear' and 'surprise' due to similar acoustic patterns.")