from pydub import AudioSegment

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format='wav')


file_name = input("Enter the name of the MP3 file (without the extension): ")

# Example usage
file_path = 'media/artists/'
mp3_file_path = file_path + file_name + '.mp3'
wav_file_path = file_path + file_name + '.wav'

convert_mp3_to_wav(mp3_file_path, wav_file_path)
