import librosa
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler


def extract_singing_features(file_path, n_mfcc=40):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=16000)

        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        significant_pitches = pitches[magnitudes > np.median(magnitudes)]
        if len(significant_pitches) > 0:
            pitch_min = np.min(significant_pitches[np.nonzero(significant_pitches)])
            pitch_max = np.max(significant_pitches)
            pitch_mean = np.mean(significant_pitches)
        else:
            pitch_min, pitch_max, pitch_mean = 0, 0, 0

        # MFCCs features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)

        # Tempo and Beat analysis
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_track = librosa.frames_to_time(beats, sr=sr)

        # Loudness dynamics
        S, phase = librosa.magphase(librosa.stft(y))
        rms = librosa.feature.rms(S=S)
        loudness_variance = np.var(rms)
        loudness_max = np.max(rms)

        return {
            'pitch_min': pitch_min,
            'pitch_max': pitch_max,
            'pitch_mean': pitch_mean,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'mfcc_max': mfcc_max,
            'mfcc_min': mfcc_min,
            'tempo': tempo,
            'beat_track': beat_track,
            'loudness_variance': loudness_variance,
            'loudness_max': loudness_max
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {}
    
# Example usage
file_path = 'media/artists/'
file_name = '박효신-야생화_vocal.wav'
features = extract_singing_features(file_path + file_name)
print(features)
file_name = '김장훈-노래만 불렀지_vocal.wav'
features = extract_singing_features(file_path + file_name)
print(features)
file_name = '버즈-나에게로 떠나는 여행_vocal.wav'
features = extract_singing_features(file_path + file_name)
print(features)
file_name = '버즈-모놀로그_vocal.wav'
features = extract_singing_features(file_path + file_name)
print(features)

