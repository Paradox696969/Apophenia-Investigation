from ImageGen import overlayImageRandom
import os
import random

dataset_path = "Visual/Normal Datasets/Flowers/"
test_path = "Visual/Stochastic Datasets/Flowers/"
num_test_samples = 20

for folder in os.listdir(dataset_path):
    for x in range(num_test_samples):
        file = random.choice(os.listdir(f"{dataset_path}{folder}/"))
        print(file)
        try:
            overlayImageRandom(224, 224, f"{dataset_path}{folder}/{file}", f"{test_path}{folder}/", file, 0.6)
        except:
            try:
                os.mkdir(f"{test_path}")
                os.mkdir(f"{test_path}{folder}/")
            except:
                pass
            overlayImageRandom(224, 224, f"{dataset_path}{folder}/{file}", f"{test_path}{folder}/", file, 0.6)