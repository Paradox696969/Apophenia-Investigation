import numpy as np
import pyaudio
import random
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

