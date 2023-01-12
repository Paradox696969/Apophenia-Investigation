from BaseImageModel import *


imgh, imgw = 64, 64
stochasLevels = ["HilbertRandom", "HilbertRandomColour", "HilbertMultiRandom", "HilbertStochastic"]

train, val, test, numclasses = loadDatasets("Numerical/Datasets/HilbertCurves/Hilbert", 32, imgh, imgw, 0.2)
model = buildModel(imgh, imgw, numclasses, data_augmentation=False)
epochs = 20
model, history, results = trainAndEvaluate(model, epochs, train, val, test)
print(f"Normal: {results}")
stats = getStats(history)

predTest(stochasLevels, model, "Numerical/Datasets/HilbertCurves/HilbertTesting/", "Numerical/Models/Results.csv", imgw, imgh, "HilbertModel", 2)
showStats(epochs, stats)

model.save("Numerical/Models/Saved/Hilbert.h5")