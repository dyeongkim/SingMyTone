import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_and_extract_mfcc(filename):
    # Load audio file
    y, sr = librosa.load(filename, sr=16000)
    # Extract MFCC features
    # MFCCs features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Average MFCCs over time
    mfcc_avg = np.mean(mfcc, axis=1)
    return mfcc_avg

def compare_voices(voice1, voice2):
    # Calculate cosine similarity between two MFCC feature vectors
    similarity = cosine_similarity([voice1], [voice2])
    return similarity[0][0]

user_voice_mfcc = load_and_extract_mfcc("media/artists/태연-To X_vocal.wav")
print(user_voice_mfcc)
singer_voice_mfcc = load_and_extract_mfcc("media/artists/버즈-나에게로 떠나는 여행_vocal.wav")

similarity_score = compare_voices(user_voice_mfcc, singer_voice_mfcc)
print(similarity_score)