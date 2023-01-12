from BaseImageModel import *


levels = ["None", "Stochastic", "Stochasticity"]
imgh, imgw = 180, 360
train, val, test, numclasses = loadDatasets("Aural/Datasets/Spectrograms", 128, imgh, imgw, 0.2)
model = buildModel(imgh, imgw, numclasses, data_augmentation=False)
epochs = 25
model, history, results = trainAndEvaluate(model, epochs, train, val, test)
print(f"Training: {results}")
stats = getStats(history)

predTest(levels, model, results, "Aural/Datasets/Spectrogram Testing Datasets/", "Aural/Models/Saved/Results.csv", imgw, imgh, "SpectrogramsModel", 2)

showStats(epochs, stats)
model.save("Aural/Models/Saved/Spectrogram.h5")

