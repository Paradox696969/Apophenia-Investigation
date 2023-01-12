import matplotlib.pyplot as plt
import os
import random
from scipy import signal
from scipy.io import wavfile
plt.axis("off")


def saveSpec(input_path, save_path):
    wavefile = wavfile.read(input_path)
    sample_rate, samples = wavefile[0], wavefile[1][:wavefile[0] * 5]
    _, _, spectrogram = signal.spectrogram(samples, sample_rate)

    
    plt.imshow(spectrogram)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)

path = "Aural/Datasets/TimeCl/"
base_save_path = "Aural/Datasets/TimeCl"

for folder in os.listdir(path):
    for sub_folder in os.listdir(f"{path}/{folder}/"):
        for file in os.listdir(f"{path}/{folder}/{sub_folder}"):
            try:
                try:
                    saveSpec(f"{path}/{folder}/{sub_folder}/{file}", f"{base_save_path}/{folder}/{sub_folder}Spec/{file[:-4]}.png")
                except:
                    os.mkdir(f"{base_save_path}/{folder}/{sub_folder}Spec/")
                    saveSpec(f"{path}/{folder}/{sub_folder}/{file}", f"{base_save_path}/{folder}/{sub_folder}Spec/{file[:-4]}.png")
                print(f"{base_save_path}/{folder}/{sub_folder}Spec/{file[:-4]}.png")
            except:
                ...
