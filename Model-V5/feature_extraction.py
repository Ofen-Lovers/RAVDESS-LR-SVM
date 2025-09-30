import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

def extract_features(file_path, n_mfcc=13):
    """
    Extracts an enhanced set of acoustic features from a single audio file.
    """
    try:
        y, sr = librosa.load(file_path, sr=None) # Load at native sample rate
        
        # Pitch
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # MFCCs and their derivatives
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # --- NEW FEATURES ---
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        # --------------------
        
        tempo = estimate_tempo_simple(y, sr)

        features = {
            'pitch_mean': np.nanmean(f0) if np.any(~np.isnan(f0)) else 0,
            'pitch_std': np.nanstd(f0) if np.any(~np.isnan(f0)) else 0,
            'rms_mean': np.mean(librosa.feature.rms(y=y)),
            'zcr_mean': np.mean(librosa.feature.zero_crossing_rate(y)),
            'tempo': tempo,
            'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'chroma_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        }
        
        # --- ADD MEANS AND STDS FOR NEW FEATURES ---
        features.update({f'spectral_contrast_{i+1}_mean': np.mean(spectral_contrast[i]) for i in range(spectral_contrast.shape[0])})
        features.update({f'spectral_contrast_{i+1}_std': np.std(spectral_contrast[i]) for i in range(spectral_contrast.shape[0])})
        features.update({f'tonnetz_{i+1}_mean': np.mean(tonnetz[i]) for i in range(tonnetz.shape[0])})
        features.update({f'tonnetz_{i+1}_std': np.std(tonnetz[i]) for i in range(tonnetz.shape[0])})
        # -------------------------------------------
        
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

def estimate_tempo_simple(y, sr):
    """Simple tempo estimation that's less likely to crash."""
    try:
        onset_env = librosa.onset.onset_detect(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return tempo[0] if isinstance(tempo, np.ndarray) else tempo
    except Exception:
        return 120.0

def load_and_extract(base_path, n_mfcc=13):
    """Loads audio files and extracts features sequentially."""
    print("Finding all .wav files...")
    all_file_paths = [os.path.join(root, file) for root, _, files in os.walk(base_path) for file in files if file.endswith('.wav')]
    if not all_file_paths:
        raise FileNotFoundError(f"No .wav files found in {base_path}.")

    print(f"Extracting features from {len(all_file_paths)} files sequentially...")
    
    features_list = []
    labels_list = []
    groups_list = [] # For speaker groups
    
    emotions = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    
    for file_path in tqdm(all_file_paths):
        try:
            filename = os.path.basename(file_path)
            parts = filename.split('-')
            if len(parts) >= 7:
                emotion_code = parts[2]
                speaker_id = int(parts[6].split('.')[0]) # Extract speaker ID
                emotion = emotions.get(emotion_code)
                if emotion:
                    features = extract_features(file_path, n_mfcc)
                    if features:
                        features_list.append(features)
                        labels_list.append(emotion)
                        groups_list.append(speaker_id) # Add speaker ID to groups list
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            continue
    
    if not features_list:
        raise ValueError("Feature extraction failed for all files.")
        
    return pd.DataFrame(features_list), pd.Series(labels_list, name="emotion"), pd.Series(groups_list, name="speaker")