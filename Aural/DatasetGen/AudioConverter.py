import os
from pydub import AudioSegment

path = "Aural/Datasets/Bird-Sounds"

audio_file_dir = os.listdir(path)
for audio_files in audio_file_dir:
    for file in os.listdir(f"{path}/{audio_files}"):
        #spliting the file into the name and the extension
        name, ext = os.path.splitext(f"{path}/{audio_files}/{file}")
        print(f"{name}{ext}")
        mp3_sound = AudioSegment.from_mp3(f"{path}/{audio_files}/{file}")
        #rename them using the old name + ".wav"
        mp3_sound.export(f"{name}.wav", format="wav")