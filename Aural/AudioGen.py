import numpy as np
import pyaudio
import random
<<<<<<< HEAD
from scipy.io.wavfile import write
from pydub import AudioSegment

p = pyaudio.PyAudio()

sound1 = pydub.AudioSegment.from_file("path")

volume = 0.25
duration = len(sound1) * 1000

samples = ()
for x in range(random.randint(0, 5)):
    fs = 44100
    f = random.randint(440, 770)

    try:
        samples += (np.sin(random.random() * random.randint(1, 9) * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)
    except:
        samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

# write(f'/home/paradox/Documents/VSCODE Projects/BTYSTE-2023/Aural/Low.wav', fs, samples)
output_bytes = (volume * samples).tobytes()
sound2 = AudioSegment(data=output_bytes, sample_width=4, framerate=44100)
=======
from scipy.io.wavfile import write, read
from pydub import AudioSegment
from pydub.playback import play
import io

p = pyaudio.PyAudio()

stochasLevel = {"1": (20, 0), "2": (18, 0.5), "3": (5, 2), "4": (-10, 50)}
stochasLevelID = "1"

sound1 = AudioSegment.from_file("C:\\Users\\joyje\\VSCODE\\BTYSTE\\Aural\\Datasets\\notes_v2\\A\\A1.wav")
sound1 += stochasLevel[stochasLevelID][0]
sound1.export("Aural\\temp.wav", format="wav")

sound = np.frombuffer(open("Aural\\temp.wav", "rb").read(), dtype=np.int16, count=len(open("Aural\\temp.wav", "rb").read())//2, offset=0)
print(sound)



volume = 0.1 * stochasLevel[stochasLevelID][1]
duration = len(sound1) / 1000
print(duration)

samples = ()
fs = 44100
for x in range(random.randint(1, 200)):
    f = random.randint(220, 1100)

    try:
        samples += (np.sin(random.random() * random.randint(0, 0) * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)
    except:
        samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

print(samples)

# write(f'Aural/Low.wav', fs, samples)
sound2_bytes = (volume * samples)
output_bytes = sound2_bytes + sound[:-(np.abs(len(sound2_bytes) - len(sound)))]
output_bytes = output_bytes.tobytes()
s = io.BytesIO(output_bytes)
sound2 = AudioSegment.from_raw(s, sample_width=sound1.sample_width, frame_rate=44100, channels=sound1.channels)
final_sound = sound2.overlay(sound1, position=0)
play(sound2)


>>>>>>> fb09d5db5 (AudioGen for research done, number seq done)

