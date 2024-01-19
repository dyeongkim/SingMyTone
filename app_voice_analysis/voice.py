import librosa
import numpy as np
from scipy.stats import mode

def extract_singing_features(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    pitch_min = np.min(pitches[np.nonzero(pitches)])
    pitch_max = np.max(pitches)
    pitch_mean = np.mean(pitches)

    # Vibrato analysis (optional, can be complex)

    # Spectral features for timbre
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_mean = np.mean(mfccs, axis=1)

    # Rhythmic features (tempo, beat tracking)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Loudness dynamics
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)
    loudness_variance = np.var(rms)

    return {
        'pitch_min': pitch_min,
        'pitch_max': pitch_max,
        'pitch_mean': pitch_mean,
        'mfcc_mean': mfcc_mean,
        'tempo': tempo,
        'loudness_variance': loudness_variance
    }

# Example usage
file_path = 'media\artists\Similar_Ballade_00001_Org.wav'
features = extract_singing_features(file_path)
print(features)
