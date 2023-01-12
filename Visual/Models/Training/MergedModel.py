from BaseModel import *
import keras
import tensorflow_datasets as tfds



keras.backend.clear_session()


imgh, imgw = 100, 100
epochs = 15
stochasLevels = ["Low-Stochas", "Med-Stochas", "High-Stochas" , "V-High-Stochas", "Pure-Stochas"]
hiddenLayers=[
                layers.Conv2D(16, 3, padding='same', activation='elu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='elu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='elu'),
                layers.MaxPooling2D(),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(128, activation='elu'),
            ]

train, val, test, numclasses = loadDatasets("Visual/Stochastic Datasets/Merged Full/", 128, imgh, imgw, 0.2)

model = buildModel(imgh, imgw, numclasses, data_augmentation=True, hiddenLayers=hiddenLayers)
model, history, normal_result = trainAndEvaluate(model, epochs, train, val, test)
stats = getStats(history)

predTest(stochasLevels, model, "Visual/Stochastic Datasets/Merged/", "Visual/Models/Results.csv", imgw, imgh, "MergedModel", 2)

showStats(epochs, stats)

model.save("Visual/Models/Saved/MergedModel.h5")