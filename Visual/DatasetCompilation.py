import os
import shutil

path = "Visual/Normal Datasets/Merged"
folders = ["train", "valid", "test"]

for data_folder in folders:
    for folder in os.listdir(f"{path}/{data_folder}/"):
        for filename in os.listdir(f"{path}/{data_folder}/{folder}"):
            try:
                shutil.move(f"{path}/{data_folder}/{folder}/{filename}", f"{path}/{folder}/{filename}")
            except:
                os.mkdir(f"{path}/{folder}/")
                shutil.move(f"{path}/{data_folder}/{folder}/{filename}", f"{path}/{folder}/{filename}")
    shutil.rmtree(f"{path}/{data_folder}/")

