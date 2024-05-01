import shutil
import os
import sqlite3
import librosa
import numpy as np
#-------------------------TEXT------------------------#
# SAVE AND COMMIT TEXT
#-----------------------------------------------------#
def clear_t(): # CLEAR TEXT
    return "", {}

def save_and_commit_text(text, prediction):
    if prediction:
        highest_prediction = max(prediction, key=prediction.get)
    else:
        highest_prediction = None

    if highest_prediction is not None:
        with sqlite3.connect('sentiment_analysis.db') as conn:
            cur = conn.cursor()
            cur.execute('''
            INSERT INTO text_analysis (text, prediction) VALUES (?, ?)
            ''', (text, highest_prediction))
            conn.commit()   
#-------------------------AUDIO------------------------#
# SAVE AND COMMIT AUDIO
#------------------------------------------------------#
def clear_a(): # CLEAR AUDIO
    return {}

def save_and_commit_audio(audio_path, prediction):
    if not prediction:
        return
    
    highest_prediction = max(prediction, key=prediction.get)
    original_ext = os.path.splitext(audio_path)[-1] 
    
    save_dir = "D:/A_Graduate Project/Full Application/Saved_Audio"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    base_filename = highest_prediction
    existing_files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
    
    count = 1
    while f"{base_filename}_{count}{original_ext}" in existing_files:
        count += 1
    
    new_filename = f"{base_filename}_{count}{original_ext}"
    saved_audio_path = os.path.join(save_dir, new_filename)
    
    shutil.copy(audio_path, saved_audio_path)
    
    with sqlite3.connect('sentiment_analysis.db') as conn:
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO audio_analysis (audio, prediction) VALUES (?, ?)
        ''', (saved_audio_path, highest_prediction))
        conn.commit()
#-------------------------------------------------------#
def preprocess_audio(filename):
    audio, sr = librosa.load(filename, sr=16000, mono=True)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=58)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_mean = np.expand_dims(mfccs_mean, axis=-1)
    mfccs_mean = np.expand_dims(mfccs_mean, axis=0)
    
    return mfccs_mean




