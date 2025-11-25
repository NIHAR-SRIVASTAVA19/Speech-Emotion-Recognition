import os
import librosa
import numpy as np
import pandas as pd
import pickle # Needed for saving/loading the DataFrame later

# -----------------------------
# Function to extract features from an audio file
# -----------------------------
def extract_features(audio_file_path):
    # Load the audio file
    # Ensure sr=None for librosa.load to prevent resampling, 
    # and duration/offset for consistent feature extraction.
    # Set a specific duration (e.g., 3 seconds) and offset to normalize input length
    y, sr = librosa.load(audio_file_path, sr=None, duration=3, offset=0.5) 
    
    # Extract MFCCs (Mel-frequency cepstral coefficients)
    # n_mfcc=40, meaning 40 coefficients per frame. np.mean(..., axis=0) averages them over time.
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    
    # Extract pitch (fundamental frequency)
    pitches = librosa.core.piptrack(y=y, sr=sr)[0]
    pitch_mean = np.mean(pitches[pitches > 0]) if pitches[pitches > 0].size > 0 else 0.0 # Handle case with no pitch detected

    # Zero Crossing Rate (how many times the signal changes sign)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    # Ensure ZCR is a single scalar. np.mean on a 1D array can result in a 1-element array.
    if isinstance(zcr, np.ndarray) and zcr.size == 1:
        zcr = zcr.item() 
    elif isinstance(zcr, np.ndarray) and zcr.size == 0: # Handle empty case
        zcr = 0.0

    # Root Mean Square (RMS) energy
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    # Ensure RMS is a single scalar
    if isinstance(rms, np.ndarray) and rms.size == 1:
        rms = rms.item()
    elif isinstance(rms, np.ndarray) and rms.size == 0: # Handle empty case
        rms = 0.0

    # Chroma feature (12-bin representation of pitch classes)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

    feature_vector = np.hstack([mfccs, [pitch_mean], [zcr], [rms], chroma, mel, contrast, tonnetz])


    # Combine all features into a single FLAT 1D vector
    # mfccs and chroma are already 1D arrays (40 and 12 elements respectively).
    # pitch_mean, zcr, rms are scalars, so we put them in a list to stack.
    # feature_vector = np.hstack([mfccs, [pitch_mean], [zcr], [rms], chroma])
    
    return feature_vector


# -----------------------------
# Emotion mapping based on RAVDESS dataset codes
# -----------------------------
emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# List to store extracted features and labels
# Each item in 'data' will now be a list: [feature1, feature2, ..., featureN, emotion_label]
data = []

# Path to dataset (Actor folders inside this directory)
parent_dataset_path = 'audio_files/audio_speech_actors_01-24'

# Check if the dataset path exists
if not os.path.exists(parent_dataset_path):
    print(f"Error: Dataset path not found: {parent_dataset_path}")
    exit() # Exit if the path is incorrect

print("Starting feature extraction from audio files...")
# -----------------------------
# Loop through all actor folders and extract features
# -----------------------------
for actor_folder in os.listdir(parent_dataset_path):
    actor_folder_path = os.path.join(parent_dataset_path, actor_folder)

    # Ensure it's a folder (not a stray file)
    if os.path.isdir(actor_folder_path):
        for audio_file in os.listdir(actor_folder_path):
            if audio_file.endswith('.wav'): # Process only .wav files
                file_path = os.path.join(actor_folder_path, audio_file)

                try:
                    # Call extract_features to get the single 1D array of all numerical features
                    features_array = extract_features(file_path)

                    # Parse emotion code and vocal channel from filename
                    parts = audio_file.split('.')[0].split('-')
                    if len(parts) < 3:
                        print(f"Warning: Skipping malformed filename: {audio_file}")
                        continue

                    vocal_channel_code = parts[1]
                    emotion_code = parts[2]
                    
                    # Exclude "song" files (vocal channel '02')
                    if vocal_channel_code == '02':
                        continue

                    if emotion_code in emotions:
                        emotion = emotions[emotion_code]
                        
                        # Convert the features_array to a list and append the emotion string
                        row = features_array.tolist() + [emotion]
                        data.append(row)
                    else:
                        print(f"Warning: Unknown emotion code '{emotion_code}' in file: {audio_file}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

print("Feature extraction complete.")
print(f"Total audio files processed: {len(data)}")

# -----------------------------
# Convert collected data to a DataFrame
# -----------------------------
# Dynamically create column names for MFCCs and Chroma features
num_mfccs = 40 # From n_mfcc in extract_features
num_chroma = 12 # Chroma has 12 bins by default
# Total number of features: 40 (mfccs) + 1 (pitch) + 1 (zcr) + 1 (rms) + 12 (chroma) = 55

# Create column names
# MFCC columns
mfcc_cols = [f'mfcc_{i}' for i in range(40)]

# Scalars
other_scalar_cols = ['pitch_mean', 'zcr', 'rms']

# Chroma columns
chroma_cols = [f'chroma_{i}' for i in range(12)]

# Mel Spectrogram columns
mel_cols = [f'mel_{i}' for i in range(128)]

# Spectral Contrast columns
contrast_cols = [f'contrast_{i}' for i in range(7)]

# Tonnetz columns
tonnetz_cols = [f'tonnetz_{i}' for i in range(6)]

# Combine all
all_feature_columns = mfcc_cols + other_scalar_cols + chroma_cols + mel_cols + contrast_cols + tonnetz_cols

# Add emotion label
df_columns = all_feature_columns + ['emotion']


# Now, create the DataFrame with the correctly flattened data and all column names
df = pd.DataFrame(data, columns=df_columns)


# --- NEW: Save the extracted features DataFrame ---
output_dir = 'processed_data'
os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist

features_df_path = os.path.join(output_dir, 'extracted_features.pkl') # Using pickle for DataFrame

print(f"\nSaving extracted features to {features_df_path}...")
df.to_pickle(features_df_path) # Use to_pickle to save the DataFrame
print("Features saved successfully!")

