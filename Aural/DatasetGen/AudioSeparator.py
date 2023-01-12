import os
import shutil

path = "Aural/Datasets/Spectrograms"

for folder in os.listdir(f"{path}"):
    for subfolder in os.listdir(f"{path}/{folder}"):
        for file in os.listdir(f"{path}/{folder}/{subfolder}"):
            shutil.move(f"{path}/{folder}/{subfolder}/{file}", f"{path}/{folder}/{subfolder}_{file}")
        os.rmdir(f"{path}/{folder}/{subfolder}")
