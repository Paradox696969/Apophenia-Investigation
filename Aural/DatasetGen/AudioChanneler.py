import os
from pydub import AudioSegment

path = "Aural/Datasets/Normal/CatsDogs/"

audio_file_dir = os.listdir(path)
for audio_files in audio_file_dir:
    for file in os.listdir(f"{path}/{audio_files}"):
        #spliting the file into the name and the extension
        name, ext = os.path.splitext(f"{path}{audio_files}/{file}")
        print(f"{path}{audio_files}/{file}")
        if ext == ".wav":
            sound = AudioSegment.from_file(f"{path}{audio_files}/{file}")
            print(sound.frame_rate)
            # #rename them using the old name + ".wav"
            sound.export(f"{name}.wav", format="wav", parameters=['-ac', '1'])
        if ext == ".png":
            os.remove(f"{name}{ext}")