from BaseModel import *
import keras
import tensorflow_datasets as tfds



keras.backend.clear_session()


imgh, imgw = 120, 120
epochs = 15
stochasLevels = ["Low-Stochas", "Med-Stochas", "High-Stochas" , "V-High-Stochas", "Pure-Stochas"]
hiddenLayers=[
                layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                layers.MaxPooling2D(),
                layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                layers.MaxPooling2D(),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(128),
            ]

train, val, test, numclasses = loadDatasets("Visual/Stochastic Datasets/Merged Stochas Evened/", 128, imgh, imgw, 0.2)

model = buildModel(imgh, imgw, numclasses, data_augmentation=False, hiddenLayers=hiddenLayers, optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001))
model, history, normal_result = trainAndEvaluate(model, epochs, train, val, test)


predTest(stochasLevels, model, "Visual/Stochastic Datasets/Merged/", "Visual/Models/Results.csv", imgw, imgh, "MergedImprovedEvenModel", 2)

stats = getStats(history)
showStats(epochs, stats)

model.save("Visual/Models/Saved/MergedImprovedEvenModel.h5")