import os
import librosa
import soundfile as sf
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = 'archive'          # The folder containing your original .wav files
TARGET_DIR = 'archive-16khz'    # The folder where the converted files will be saved
TARGET_SR = 16000               # The target sample rate (16,000 Hz)

def convert_audio_files():
    """
    Finds all .wav files in the source directory, converts them to the target
    sample rate, and saves them in the target directory, preserving the
    original folder structure.
    """
    print("--- Audio Conversion to 16kHz ---")
    print(f"Source directory: '{SOURCE_DIR}'")
    print(f"Target directory: '{TARGET_DIR}'")
    print("-" * 35)

    # --- Step 1: Find all .wav files in the source directory ---
    source_files = []
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith('.wav'):
                source_path = os.path.join(root, file)
                source_files.append(source_path)

    if not source_files:
        print(f"Error: No .wav files found in '{SOURCE_DIR}'. Please check the path.")
        return

    print(f"Found {len(source_files)} .wav files to convert.")

    # --- Step 2: Create the target root directory if it doesn't exist ---
    os.makedirs(TARGET_DIR, exist_ok=True)

    # --- Step 3: Loop through files, convert, and save ---
    # Using tqdm for a nice progress bar
    for source_path in tqdm(source_files, desc="Converting files"):
        try:
            # Determine the corresponding path in the target directory
            relative_path = os.path.relpath(source_path, SOURCE_DIR)
            target_path = os.path.join(TARGET_DIR, relative_path)

            # Create the subdirectory in the target folder if it doesn't exist
            target_subdir = os.path.dirname(target_path)
            os.makedirs(target_subdir, exist_ok=True)

            # Load the audio file and resample it to the target sample rate
            # librosa.load() handles the resampling automatically when sr is specified
            y, sr = librosa.load(source_path, sr=TARGET_SR)

            # Save the resampled audio to the new location
            sf.write(target_path, y, TARGET_SR)

        except Exception as e:
            print(f"\nWarning: Failed to process '{source_path}'. Error: {e}")
            # Continue to the next file even if one fails
            continue

    print("\n--- Conversion Complete! ---")
    print(f"All convertible files have been processed and saved in '{TARGET_DIR}'.")


# --- Main execution block ---
if __name__ == "__main__":
    # This ensures the function runs only when the script is executed directly
    convert_audio_files()