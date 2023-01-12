import numpy as np
import random
from pydub import AudioSegment
import io
global logs
logs = ""
def stochastizeAudio(
    audioFile="Aural/Datasets/Normal/Bird-Sounds/XC4977.mp3", 
    output="/home/paradox/Documents/VSCODE Projects/BTYSTE-2023/Aural/Research-Tests/Stochastic/5.wav",
    volume1=0.001,
    volume2=1
    ):
    try:
        audioFile2 = "/home/paradox/Documents/VSCODE Projects/BTYSTE-2023/Aural/temp.wav"
        sound1 = AudioSegment.from_file(audioFile)
        sound1.export(audioFile2, format="wav")

        sound = np.frombuffer(open(audioFile2, "rb").read(), dtype=np.float16)


        duration = len(sound1) / 1000
        print(duration)

        samples = ()
        fs = sound1.frame_rate
        for x in range(random.randint(1, 50)):
            f = random.randint(440, 2200)

            try:
                samples += (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float16)
            except:
                samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float16)


        # write(f'Aural/Low.wav', fs, samples)
        sound2_bytes = (volume1 * samples)
        length = len(sound[:-(np.abs(len(sound2_bytes) - len(sound)))])
        print(sound2_bytes)
        print(length)
        print(audioFile)

        output_bytes = sound2_bytes[:length] + (volume2 * sound[:length])
        output_bytes = output_bytes.tobytes()
        s = io.BytesIO(output_bytes)
        final_sound_full = AudioSegment.from_raw(s, sample_width=sound1.sample_width, frame_rate=fs, channels=sound1.channels)
        final_sound = final_sound_full[:len(final_sound_full)]
        final_sound.export(output, format="wav")
    except Exception as e:
        global logs
        logs += f"{e}\n\n"

    print(logs)

