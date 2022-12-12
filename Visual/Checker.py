from PIL import Image
import os

path = "Visual/Normal Datasets/Animals"

for folder in os.listdir(path):
    for filename in os.listdir(f"{path}/{folder}"):
        print(folder, filename)
        Image.open(f"{path}/{folder}/{filename}")