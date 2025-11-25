import librosa
import numpy as np

# Define allowed extensions globally for file checks in Flask
# *** CRITICAL: Ensure 'webm' is included for live recording ***
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'webm'}

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------
# Function to extract features from an audio file 
# (Includes all features: MFCCs, pitch, ZCR, RMS, Chroma, Mel, Contrast, Tonnetz)
# -----------------------------
def extract_features(audio_file_path):
    """
    Extracts features required by the SVM model.
    MUST match the features used during training.
    """
    try:
        # Librosa loads various formats, including webm
        y, sr = librosa.load(audio_file_path, sr=None, duration=3, offset=0.5) 
        
        # Check if the audio data is empty (can happen with failed recordings)
        if y.size == 0:
            print(f"DEBUG: Empty audio data loaded from {audio_file_path}")
            return None

        # Extract MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        
        # Extract pitch (fundamental frequency)
        pitches = librosa.core.piptrack(y=y, sr=sr)[0]
        pitch_mean = np.mean(pitches[pitches > 0]) if pitches[pitches > 0].size > 0 else 0.0

        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
        zcr = zcr.item() if isinstance(zcr, np.ndarray) and zcr.size == 1 else 0.0

        # Root Mean Square (RMS) energy
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        rms = rms.item() if isinstance(rms, np.ndarray) and rms.size == 1 else 0.0

        # Chroma feature
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        
        # Spectral Contrast
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        
        # Tonnetz
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

        # Combine all features into a single FLAT 1D vector
        feature_vector = np.hstack([mfccs, [pitch_mean], [zcr], [rms], chroma, mel, contrast, tonnetz])
        
        print(f"DEBUG: Extracted {feature_vector.shape[0]} features.")
        return feature_vector

    except Exception as e:
        print(f"ERROR: Feature extraction failed for {audio_file_path}: {e}")
        return None
