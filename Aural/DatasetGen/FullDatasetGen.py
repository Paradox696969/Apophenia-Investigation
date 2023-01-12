from AudioStochastizer import stochastizeAudio
import os


path = "Aural/Datasets/Normal"
save_path = "Aural/Datasets/TimeCl"
randomness = {"None": (0, 1), "Stochastic": (0.5, 0.5), "Stochasticity": (1, 0)}
randoms = ["None", "Stochastic", "Stochasticity"]
num_test_samples = [200, 400, 600, 800, 1000, 2000]
print(sorted(os.listdir(f"{path}")))
listed = [sorted(os.listdir(f"{path}"))[0]]
for folder in listed:
    files = os.listdir(f"{path}/{folder}")
    for num_test_sample in num_test_samples:
        for level in randoms:
            for i in range(num_test_sample):
                discriminant = i / len(os.listdir(f"{path}/{folder}/"))
                file = files[i - len(os.listdir(f"{path}/{folder}/")) * int(discriminant)]
                try:
                    os.mkdir(f"{save_path}/{level}_{num_test_sample}/{folder}")
                    stochastizeAudio(f"{path}/{folder}/{file}", f"{save_path}/{level}_{num_test_sample}/{folder}/{file[:-4]}_{discriminant}.wav", randomness[level][0], randomness[level][1])
                except FileExistsError:
                    stochastizeAudio(f"{path}/{folder}/{file}", f"{save_path}/{level}_{num_test_sample}/{folder}/{file[:-4]}_{discriminant}.wav", randomness[level][0], randomness[level][1])
                except:
                    os.mkdir(f"{save_path}/{level}_{num_test_sample}")
                    os.mkdir(f"{save_path}/{level}_{num_test_sample}/{folder}")
                    stochastizeAudio(f"{path}/{folder}/{file}", f"{save_path}/{level}_{num_test_sample}/{folder}/{file[:-4]}_{discriminant}.wav", randomness[level][0], randomness[level][1])
                
                