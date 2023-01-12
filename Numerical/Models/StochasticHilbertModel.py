from BaseImageModel import *


imgh, imgw = 64, 64
stochasLevels = ["HilbertRandom", "HilbertRandomColour", "HilbertMultiRandom", "HilbertStochastic"]
hiddenLayers=[
                layers.Conv2D(16, 3, padding='same', activation='elu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='elu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='elu'),
                layers.MaxPooling2D(),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(128),
            ]
train, val, test, numclasses = loadDatasets("Numerical/Datasets/HilbertCurves/HilbertStochas", 32, imgh, imgw, 0.2)
model = buildModel(imgh, imgw, numclasses, data_augmentation=True, hiddenLayers=hiddenLayers, optimizer=tf.keras.optimizers.Nadam(0.1))
epochs = 20
model, history, results = trainAndEvaluate(model, epochs, train, val, test)
print(f"Normal: {results}")
stats = getStats(history)

predTest(stochasLevels, model, "Numerical/Datasets/HilbertCurves/HilbertTesting/", "Numerical/Models/Results.csv", imgw, imgh, "StochasticHilbertModel", 2)
showStats(epochs, stats)

model.save("Numerical/Models/Saved/HilbertStochastic.h5")