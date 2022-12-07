import numpy as np
import random
from pydub import AudioSegment
import io

audioFile = "/home/paradox/Documents/VSCODE Projects/BTYSTE-2023/Aural/Datasets/Bird-Sounds/xeno-canto-ca-nv/XC7426.mp3"
audioFile2 = "/home/paradox/Documents/VSCODE Projects/BTYSTE-2023/Aural/temp.wav"
output = "/home/paradox/Documents/VSCODE Projects/BTYSTE-2023/Aural/Research-Tests/Stochastic/5.wav"
sound1 = AudioSegment.from_file(audioFile)
sound1.export(audioFile2, format="wav")

sound = np.frombuffer(open(audioFile2, "rb").read(), dtype=np.float16)



volume1 = 0.001
volume2 = 1
duration = len(sound1) / 1000
print(duration)

samples = ()
fs = sound1.frame_rate
for x in range(random.randint(1, 200)):
    f = random.randint(440, 770)

    try:
        samples += (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float16)
    except:
        samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float16)


# write(f'Aural/Low.wav', fs, samples)
sound2_bytes = (volume1 * samples)
print(sound2_bytes)
print(volume2 * sound[:-(np.abs(len(sound2_bytes) - len(sound)))])

output_bytes = sound2_bytes + (volume2 * sound[:-(np.abs(len(sound2_bytes) - len(sound)))])
output_bytes = output_bytes.tobytes()
s = io.BytesIO(output_bytes)
final_sound_full = AudioSegment.from_raw(s, sample_width=sound1.sample_width, frame_rate=fs, channels=sound1.channels)
final_sound = final_sound_full[:len(final_sound_full)]
final_sound.export(output, format="wav")


