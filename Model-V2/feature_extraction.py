import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

def extract_features(file_path, n_mfcc=13):
    """
    Extracts a comprehensive set of acoustic features from a single audio file.
    Includes tempo with safe error handling.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Pitch
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # MFCCs and their derivatives (Deltas)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
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
    """
    Simple tempo estimation that's less likely to crash.
    Uses spectral flux for basic tempo estimation.
    """
    try:
        # Use a simpler approach - spectral flux peaks
        hop_length = 512
        frame_length = 2048
        
        # Compute spectral flux
        stft = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=frame_length))
        spectral_flux = np.sum(np.diff(stft, axis=1)**2, axis=0)
        
        # Find peaks in spectral flux
        peaks = librosa.util.peak_pick(spectral_flux, pre_max=3, post_max=3, 
                                     pre_avg=3, post_avg=5, delta=0.5, wait=10)
        
        if len(peaks) > 1:
            # Calculate average time between peaks
            peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
            intervals = np.diff(peak_times)
            if len(intervals) > 0:
                avg_interval = np.mean(intervals)
                tempo = 60.0 / avg_interval if avg_interval > 0 else 120.0
                return min(max(tempo, 60.0), 240.0)  # Constrain to reasonable range
        
        return 120.0  # Default tempo
        
    except Exception as e:
        print(f"Simple tempo estimation failed: {e}")
        return 120.0  # Fallback tempo

def load_and_extract(base_path, n_mfcc=13):
    """Loads audio files and extracts features sequentially (more stable)."""
    print("Finding all .wav files...")
    all_file_paths = [os.path.join(root, file) for root, _, files in os.walk(base_path) for file in files if file.endswith('.wav')]
    if not all_file_paths:
        raise FileNotFoundError(f"No .wav files found in {base_path}.")

    print(f"Extracting features from {len(all_file_paths)} files sequentially...")
    
    features_list = []
    labels_list = []
    
    emotions = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    
    for file_path in tqdm(all_file_paths):
        try:
            filename = os.path.basename(file_path)
            parts = filename.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = emotions.get(emotion_code)
                if emotion:
                    features = extract_features(file_path, n_mfcc)
                    if features:
                        features_list.append(features)
                        labels_list.append(emotion)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            continue
    
    if not features_list:
        raise ValueError("Feature extraction failed for all files.")
        
    return pd.DataFrame(features_list), pd.Series(labels_list, name="emotion")