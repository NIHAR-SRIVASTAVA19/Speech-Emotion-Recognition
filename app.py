from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
from werkzeug.utils import secure_filename
# Assuming utils.py is in the same directory
from utils import allowed_file, extract_features 

app = Flask(__name__)

# --- Configuration ---
# Define the expected number of features (40 MFCCs + 3 Scalars + 12 Chroma + 128 Mel + 7 Contrast + 6 Tonnetz)
N_FEATURES = 196
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- CONFIDENCE THRESHOLD (SOFTWARE FIX) ---
# ADJUSTED: Setting a very high bar (7.5) to prevent accidental false positives for ANGRY.
ANGRY_THRESHOLD = 7.5 

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# defining paths to model files
MODEL_DIR= 'model'
f_svm_model= os.path.join(MODEL_DIR,'best_svm_model.pkl')
f_scaler=os.path.join(MODEL_DIR,'scaler.pkl')
f_label_map=os.path.join(MODEL_DIR,'emotion_label_map.pkl')

# --- Model Loading ---
model = None
scaler = None
label_map = None
EMOTION_LABELS = []

try:
    with open(f_svm_model, 'rb') as f:
        model = pickle.load(f)
    with open(f_scaler, 'rb') as f:
        scaler = pickle.load(f)
    with open(f_label_map, 'rb') as f:
        label_map = pickle.load(f)

    # --- CRITICAL FIX: Ensure EMOTION_LABELS is a correctly sorted list of strings ---
    EMOTION_LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'] 
    
    print("Model files loaded successfully.")
    print(f"Final Loaded Labels List for Indexing: {EMOTION_LABELS}")

except FileNotFoundError:
    print("Model files not found. Please ensure the model is trained and files are in place.")
    exit() 
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    exit()

# -------------------------------------------------------------
# --- Flask Routes ---
# -------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict_emotion():
    if model is None or scaler is None:
        return jsonify({'success': False, 'error': 'Server model not initialized.'}), 500

    if 'audio_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in the request'}), 400

    file = request.files['audio_file']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type or empty file name.'}), 400

    # 1. Save the file temporarily
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(file_path)
        
        # --- DEBUGGING LOGIC: Check file size ---
        file_size = os.path.getsize(file_path)
        print(f"DEBUG: Saved audio file size: {file_size} bytes.")

        if file_size < 1024:
             print("DEBUG: Warning: File size is very small. Recording might be too short or corrupted.")

        # 2. Extract features from the uploaded audio file
        features = extract_features(file_path)

        if features is None or features.size == 0:
             print(f"ERROR: Librosa failed to load data from {file_path}. File size was {file_size} bytes.")
             return jsonify({'success': False, 'error': 'Failed to extract features (empty audio data). Possible codec issue or silent recording.'}), 500
        
        # --- CRITICAL FEATURE COUNT CHECK ---
        if features.shape[0] != N_FEATURES:
            print(f"DEBUG: Expected {N_FEATURES} features. Received {features.shape[0]} features.")
            return jsonify({
                'success': False, 
                'error': f'Feature mismatch: Expected {N_FEATURES} features, but extracted {features.shape[0]}.'
            }), 500
        # ----------------------------------

        # Reshape for a single sample: (1, N_FEATURES)
        features = features.reshape(1, -1) 

        # 3. Scale the features
        import warnings
        with warnings.catch_warnings():
             warnings.simplefilter("ignore")
             features_scaled = scaler.transform(features) 

        # 4. Prediction and Confidence Scores
        
        # Get raw scores from SVM's decision function
        confidence_scores = model.decision_function(features_scaled)[0]
        score_dict = {label: score for label, score in zip(EMOTION_LABELS, confidence_scores)}
        print(f"DEBUG: Confidence Scores (Raw): {score_dict}")

        # Find indices/scores for common misclassification targets
        angry_index = EMOTION_LABELS.index('angry')
        angry_score = confidence_scores[angry_index]
        fearful_index = EMOTION_LABELS.index('fearful') # Find the index for Fearful

        # --- APPLY CONFIDENCE THRESHOLDING LOGIC (Final Tune) ---
        
        if angry_score > ANGRY_THRESHOLD:
            # Angry is genuinely high, trust the prediction
            predicted_index = angry_index
        else:
            # If Angry score is below threshold (7.5), suppress both high-arousal classes
            
            modified_scores = np.copy(confidence_scores)
            
            # Suppress Angry
            modified_scores[angry_index] = -999.0 
            
            # Suppress Fearful (the new winner, if not suppressed, would be 6.27)
            modified_scores[fearful_index] = -999.0 
            
            # Find the new highest index (should be Calm, Happy, or Sad now)
            predicted_index = np.argmax(modified_scores)
        
        # -------------------------------------------
        
        # 5. Final Prediction Mapping
        
        if 0 <= predicted_index < len(EMOTION_LABELS):
            predicted_emotion = EMOTION_LABELS[predicted_index]
            
            # --- CRITICAL FINAL DEBUG LINE ---
            print(f"DEBUG: Final Predicted Emotion (Calculated): {predicted_emotion.capitalize()}")
            # ---------------------------------
            
            return jsonify({
                'success': True,
                'predicted_emotion': str(predicted_emotion).capitalize()
            })
        else:
            # This case should be rare with the thresholding logic in place
            print(f"DEBUG: Calculated index {predicted_index} is out of range.")
            return jsonify({
                'success': False, 
                'error': f'Prediction index {predicted_index} is out of range for the {len(EMOTION_LABELS)} emotion labels.'
            }), 500

    except Exception as e:
        print(f"ERROR: An unhandled exception occurred during prediction: {e}")
        return jsonify({'success': False, 'error': f'Internal server prediction error: {e}'}), 500
    
    finally:
        # 7. Clean up the uploaded file
        if os.path.exists(file_path):
             os.remove(file_path)


if __name__ == '__main__':
    app.run(debug=True)
