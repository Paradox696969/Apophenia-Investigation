from BaseModel import *


imgh, imgw = 224, 224
train, val, test, numclasses = loadDatasets("Visual/Normal Datasets/Animals", 32, imgh, imgw, 0.2)
model = buildModel(imgh, imgw, numclasses)
epochs = 50
model, history, results = trainAndEvaluate(model, epochs, train, val, test)
print(f"Normal: {results}")
stats = getStats(history)
pred_ds = loadPredDs("Visual/Stochastic Datasets/Animals/", 32, imgh, imgw)
print(f"Stochastic: {predEval(pred_ds, model)[1]}")
showStats(epochs, stats)