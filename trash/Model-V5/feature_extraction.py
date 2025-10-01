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
        # Debug: Print which file we're processing
        print(f"Processing: {os.path.basename(file_path)}")
        
        # Load audio with error handling
        try:
            y, sr = librosa.load(file_path, sr=22050)  # Use consistent sample rate
            print(f"  Loaded: {len(y)} samples at {sr}Hz")
        except Exception as e:
            print(f"  ERROR loading audio: {e}")
            return None
        
        # Check if audio is not empty
        if len(y) == 0:
            print("  ERROR: Empty audio file")
            return None
            
        # Basic audio stats for debugging
        print(f"  Duration: {len(y)/sr:.2f}s, RMS: {np.sqrt(np.mean(y**2)):.4f}")
        
        # Pitch extraction with error handling
        try:
            f0, _, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)  # Reduced range for stability
            pitch_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0
            pitch_std = np.nanstd(f0) if np.any(~np.isnan(f0)) else 0
            print(f"  Pitch: mean={pitch_mean:.2f}, std={pitch_std:.2f}")
        except Exception as e:
            print(f"  ERROR in pitch extraction: {e}")
            pitch_mean, pitch_std = 0, 0
        
        # MFCCs - most reliable features
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
            print(f"  MFCCs: {mfccs.shape}")
        except Exception as e:
            print(f"  ERROR in MFCC extraction: {e}")
            return None
        
        # Simplified feature set for stability
        features = {
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'rms_mean': np.mean(librosa.feature.rms(y=y)),
            'zcr_mean': np.mean(librosa.feature.zero_crossing_rate(y)),
            'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        }
        
        # Add only MFCC means (skip std and derivatives for now)
        for i in range(min(5, n_mfcc)):  # Start with just 5 MFCCs
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        
        print(f"  Extracted {len(features)} features successfully")
        return features
        
    except Exception as e:
        print(f"CRITICAL ERROR processing {os.path.basename(file_path)}: {e}")
        return None

def load_and_extract_safe(base_path, n_mfcc=13, max_files=10):
    """
    Safe version that processes only a few files first for testing.
    """
    print("Finding all .wav files...")
    all_file_paths = []
    
    # Recursively find all .wav files
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                all_file_paths.append(full_path)
    
    if not all_file_paths:
        raise FileNotFoundError(f"No .wav files found in {base_path}.")
    
    print(f"Found {len(all_file_paths)} .wav files")
    
    # Test with first few files only
    if len(all_file_paths) > max_files:
        print(f"⚠️  TEST MODE: Processing first {max_files} files only")
        all_file_paths = all_file_paths[:max_files]
    
    features_list = []
    labels_list = []
    groups_list = []
    
    emotions = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    
    successful = 0
    failed = 0
    
    print(f"\nStarting feature extraction for {len(all_file_paths)} files...")
    
    for file_path in all_file_paths:
        try:
            filename = os.path.basename(file_path)
            print(f"\n--- Processing: {filename} ---")
            
            parts = filename.split('-')
            if len(parts) < 2:
                print(f"  SKIP: Invalid filename format")
                failed += 1
                continue
                
            # RAVDESS filename format: 03-01-06-01-02-01-12.wav
            if len(parts) >= 7:
                emotion_code = parts[2]
                speaker_id = parts[6].split('.')[0]
                
                # Validate speaker ID is numeric
                if not speaker_id.isdigit():
                    print(f"  SKIP: Invalid speaker ID: {speaker_id}")
                    failed += 1
                    continue
                    
                speaker_id = int(speaker_id)
                emotion = emotions.get(emotion_code)
                
                if emotion:
                    features = extract_features(file_path, n_mfcc)
                    if features:
                        features_list.append(features)
                        labels_list.append(emotion)
                        groups_list.append(speaker_id)
                        successful += 1
                        print(f"  ✅ SUCCESS: {emotion} - Speaker {speaker_id}")
                    else:
                        failed += 1
                        print(f"  ❌ FEATURE EXTRACTION FAILED")
                else:
                    failed += 1
                    print(f"  SKIP: Unknown emotion code: {emotion_code}")
            else:
                failed += 1
                print(f"  SKIP: Unexpected filename format")
                
        except Exception as e:
            failed += 1
            print(f"❌ CRITICAL ERROR processing {filename}: {e}")
            continue
    
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total processed: {successful + failed}")
    
    if not features_list:
        raise ValueError("Feature extraction failed for all files. Check audio files and paths.")
    
    # Create DataFrame with proper error handling
    try:
        features_df = pd.DataFrame(features_list)
        print(f"Created features DataFrame with {len(features_df)} rows and {len(features_df.columns)} columns")
        
        # Fill any NaN values
        if features_df.isnull().any().any():
            print("WARNING: NaN values found, filling with 0")
            features_df = features_df.fillna(0)
            
        return features_df, pd.Series(labels_list, name="emotion"), pd.Series(groups_list, name="speaker")
        
    except Exception as e:
        print(f"ERROR creating DataFrame: {e}")
        raise