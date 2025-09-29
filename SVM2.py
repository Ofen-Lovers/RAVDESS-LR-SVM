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

# %% Feature Extraction
# NOTE: This section extracts features from audio files and saves them to a CSV.
# This can be time-consuming. If the CSV file already exists, it will be loaded directly.

dataset_path = "dataset_16k"
csv_filename = "ravdess_full_features.csv"

if os.path.exists(csv_filename):
    print(f"Loading features from {csv_filename}")
    df = pd.read_csv(csv_filename)
else:
    print("Extracting features from dataset...")
    features_list = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if not file.endswith(".wav"):
                continue
            
            file_path = os.path.join(root, file)
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
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            # Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            
            # Pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches)
            
            # Flatten features
            features = [
                np.mean(zcr), np.mean(rms), duration, tempo,
                np.mean(spectral_centroid), np.mean(spectral_bandwidth),
                np.mean(spectral_contrast), np.mean(spectral_flatness),
                pitch_mean
            ]
            features += [np.mean(mfcc[i]) for i in range(13)]
            features += [np.mean(delta_mfcc[i]) for i in range(13)]
            features += [np.mean(delta2_mfcc[i]) for i in range(13)]
            features += [np.mean(chroma[i]) for i in range(chroma.shape[0])]
            features += [np.mean(tonnetz[i]) for i in range(tonnetz.shape[0])]
            
            # Emotion label
            emotion = int(file.split("-")[2])
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
    print(f"Full features saved to {csv_filename}")

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
# Linear SVM
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)

# RBF SVM
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)

# %% Model Evaluation
# Evaluation function
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

print("Comparative Evaluation Table:")
print(results)

# %% Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_rbf)  # Using best model (RBF)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (RBF SVM)")
plt.show()

# %% Classification Report and Analysis
# Detailed classification report
print(classification_report(y_test, y_pred_rbf, zero_division=0))

# Short analysis
# Usually in RAVDESS, emotions like "calm" and "neutral" are harder to classify,
# as well as "fear" and "surprise" due to similar acoustic patterns.