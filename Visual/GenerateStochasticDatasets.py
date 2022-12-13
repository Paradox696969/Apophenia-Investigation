from ImageGen import overlayImageRandom
import os
import random

dataset_path = "Visual/Normal Datasets/Flowers/"
test_path = "Visual/Stochastic Datasets/Flowers/"
stochasLevel = "Pure-Stochas"
num_test_samples = 20
try:
    os.mkdir(f"{test_path}")
    os.mkdir(f"{test_path}{stochasLevel}/")
except:
    os.mkdir(f"{test_path}{stochasLevel}/")

for folder in os.listdir(dataset_path):
    for x in range(num_test_samples):
        file = random.choice(os.listdir(f"{dataset_path}{folder}/"))
        print(file)
        try:
            os.mkdir(f"{test_path}{stochasLevel}/{folder}/")
            overlayImageRandom(224, 224, f"{dataset_path}{folder}/{file}", f"{test_path}{folder}/", file, 0)
        except:
            overlayImageRandom(224, 224, f"{dataset_path}{folder}/{file}", f"{test_path}{stochasLevel}/{folder}/", file, 0)