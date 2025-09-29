import os
import numpy as np
import pandas as pd
import librosa
import joblib
from tqdm import tqdm

def extract_features(file_path, n_mfcc=13):
    """
    Extracts a comprehensive set of acoustic features from a single audio file.
    (Tempo has been removed to ensure stability with parallel processing).
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Pitch
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # MFCCs and their derivatives (Deltas)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # --- Tempo calculation has been removed ---
        # tempo, _ = librosa.beat.beat_track(y=y, sr=sr) # <-- THIS LINE IS REMOVED

        features = {
            'pitch_mean': np.nanmean(f0) if np.any(~np.isnan(f0)) else 0,
            'pitch_std': np.nanstd(f0) if np.any(~np.isnan(f0)) else 0,
            'rms_mean': np.mean(librosa.feature.rms(y=y)),
            'zcr_mean': np.mean(librosa.feature.zero_crossing_rate(y)),
            # 'tempo': tempo, # <-- THIS LINE IS REMOVED
            'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'chroma_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        }
        
        for i in range(n_mfcc):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            features[f'delta_mfcc_{i+1}_mean'] = np.mean(delta_mfccs[i])
            features[f'delta_mfcc_{i+1}_std'] = np.std(delta_mfccs[i])
            features[f'delta2_mfcc_{i+1}_mean'] = np.mean(delta2_mfccs[i])
            features[f'delta2_mfcc_{i+1}_std'] = np.std(delta2_mfccs[i])
            
        return features
        
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

def _process_file(file_path, n_mfcc):
    """Internal helper for parallel processing."""
    emotions = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    filename = os.path.basename(file_path)
    parts = filename.split('-')
    if len(parts) >= 3:
        emotion_code = parts[2]
        emotion = emotions.get(emotion_code)
        if emotion:
            features = extract_features(file_path, n_mfcc)
            if features:
                return features, emotion
    return None, None

def load_and_extract(base_path, n_mfcc=13):
    """Loads audio files and extracts features in parallel."""
    print("Finding all .wav files...")
    all_file_paths = [os.path.join(root, file) for root, _, files in os.walk(base_path) for file in files if file.endswith('.wav')]
    if not all_file_paths:
        raise FileNotFoundError(f"No .wav files found in {base_path}.")

    print(f"Extracting features from {len(all_file_paths)} files using parallel processing...")
    results = joblib.Parallel(n_jobs=-1)(joblib.delayed(_process_file)(fp, n_mfcc) for fp in tqdm(all_file_paths))
    
    processed_results = [res for res in results if res[0] is not None]
    if not processed_results:
        raise ValueError("Feature extraction failed for all files.")
        
    features_list, labels_list = zip(*processed_results)
    return pd.DataFrame(features_list), pd.Series(labels_list, name="emotion")