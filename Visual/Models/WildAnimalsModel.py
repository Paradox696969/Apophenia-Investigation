from BaseModel import *


imgh, imgw = 224, 244
train, val, test, numclasses = loadDatasets("Visual/Normal Datasets/Animals", 32, imgh, imgw, 0.2)
model = buildModel(imgh, imgw, numclasses)
epochs = 10
model, history, results = trainAndEvaluate(model, epochs, train, val, test)
print(results)
stats = getStats(history)
showStats(epochs, stats)