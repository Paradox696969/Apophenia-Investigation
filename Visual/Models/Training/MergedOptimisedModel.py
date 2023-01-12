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

train, val, test, numclasses = loadDatasets("Visual/Stochastic Datasets/Merged Stochas/", 128, imgh, imgw, 0.2)

model = buildModel(imgh, imgw, numclasses, data_augmentation=True, hiddenLayers=hiddenLayers, optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001))
model, history, normal_result = trainAndEvaluate(model, epochs, train, val, test)


predTest(stochasLevels, model, "Visual/Stochastic Datasets/Merged/", "Visual/Models/Results.csv", imgw, imgh, "MergedImprovedModel", 2)

stats = getStats(history)
showStats(epochs, stats)

model.save("Visual/Models/Saved/MergedImprovedModel.h5")