from BaseModel import *


imgh, imgw = 180, 180
train, val, test, numclasses = loadDatasets("Visual/Normal Datasets/Flowers", 32, imgh, imgw, 0.2)
model = buildModel(imgh, imgw, numclasses, data_augmentation=True)
epochs = 50
model, history, results = trainAndEvaluate(model, epochs, train, val, test)
print(f"Normal: {results}")
stats = getStats(history)
pred_ds = loadPredDs("Visual/Stochastic Datasets/Flowers/", 32, imgh, imgw)
print(f"Stochastic: {predEval(pred_ds, model)[1]}")
showStats(epochs, stats)