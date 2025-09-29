# SVM.py

# %% Imports
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
import librosa.feature
import ast

# %% Feature Extraction
# NOTE: This script assumes audio files have been pre-converted to 16kHz.
dataset_path = "archive-16khz-v2"  # Path to your 16kHz audio files
csv_path = "ravdess_features.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The directory '{dataset_path}' was not found. Please ensure your 16kHz audio files are in this folder.")

if os.path.exists(csv_path):
    print(f"'{csv_path}' already exists. Loading features.")
    df = pd.read_csv(csv_path)
else:
    print("Extracting features: Pitch, RMS, ZCR, Tempo, Duration, Delta MFCC")
    def extract_features(path):
        y, sr = librosa.load(path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        pitch_mean = np.nanmean(f0)
        rms_mean = np.mean(librosa.feature.rms(y=y))
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mfcc_mean = np.mean(delta_mfcc, axis=1)

        return {
            "pitch_mean": pitch_mean,
            "rms_mean": rms_mean,
            "zcr_mean": zcr_mean,
            "tempo_bpm": tempo,
            "duration_s": duration,
            **{f"delta_mfcc{i+1}": float(delta_mfcc_mean[i]) for i in range(13)}
        }

    all_feats = []
    for root, _, files in os.walk(dataset_path):
        for f in tqdm(files, desc=f"Processing files in {root}"):
            if f.lower().endswith(".wav"):
                file_path = os.path.join(root, f)
                feats = extract_features(file_path)
                feats["filename"] = f
                all_feats.append(feats)

    df = pd.DataFrame(all_feats)
    df.to_csv(csv_path, index=False)
    print(f"Features saved to {csv_path}")

# %% Data Cleaning and Labeling
def get_emotion_from_filename(fname):
    parts = fname.split("-")
    emotion_id = int(parts[2])
    mapping = {
        1: "neutral", 2: "calm", 3: "happy", 4: "sad",
        5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
    }
    return mapping.get(emotion_id)

df['emotion'] = df['filename'].apply(get_emotion_from_filename)

def clean_tempo(val):
    if isinstance(val, str):
        try:
            v = ast.literal_eval(val)
            if isinstance(v, (list, tuple, np.ndarray)):
                return float(v[0]) if len(v) else np.nan
            return float(v)
        except (ValueError, SyntaxError):
            return np.nan
    return val

df['tempo_bpm'] = df['tempo_bpm'].apply(clean_tempo)
df['tempo_bpm'] = pd.to_numeric(df['tempo_bpm'], errors='coerce').fillna(0)
print("Cleaned 'tempo_bpm' column data type:", df['tempo_bpm'].dtype)


# %% Train-Test Split
from sklearn.model_selection import train_test_split

X = df.drop(columns=['emotion', 'filename'])
y = df['emotion']
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# %% SVM Pipeline Training
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42
    ))
])

svm_clf.fit(X_train, y_train)
print("SVM model trained.")

# %% Model Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=svm_clf.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=svm_clf.classes_,
            yticklabels=svm_clf.classes_,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# %% SHAP Feature Importance (Optional and Slow)
# NOTE: The SHAP analysis can be very slow.
try:
    import shap
    print("\nStarting SHAP analysis (this may take a while)...")
    
    X_train_arr = X_train.to_numpy()
    X_test_arr  = X_test.to_numpy()
    
    predict_fn = lambda data: svm_clf.predict_proba(data)
    
    background = shap.sample(X_train_arr, 50, random_state=42)
    explainer = shap.KernelExplainer(predict_fn, background)
    
    test_sample = shap.sample(X_test_arr, 20, random_state=42)
    shap_values = explainer.shap_values(test_sample)
    
    shap.summary_plot(
        shap_values,
        test_sample,
        feature_names=X_train.columns,
        max_display=len(X_train.columns),
        plot_type="bar",
        show=True
    )
except ImportError:
    print("\nSHAP library not found. Skipping SHAP analysis.")
except Exception as e:
    print(f"\nAn error occurred during SHAP analysis: {e}")
    print("Skipping SHAP plot.")