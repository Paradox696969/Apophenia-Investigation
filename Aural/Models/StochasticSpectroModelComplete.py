from BaseImageModel import *

hiddenLayers=[
                        layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                        layers.MaxPooling2D(),
                        layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                        layers.MaxPooling2D(),
                        layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                        layers.MaxPooling2D(),
                        layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
                        layers.MaxPooling2D(),
                        layers.Dropout(0.2),
                        layers.Flatten(),
                        layers.Dense(128, activation='relu')
                    ]
levels = ["None", "Stochastic", "Stochasticity"]
imgh, imgw = 180, 360
train, val, test, numclasses = loadDatasets("Aural/Datasets/StochasticSpectrograms", 128, imgh, imgw, 0.2)
model = buildModel(imgh, imgw, numclasses, data_augmentation=False, hiddenLayers=hiddenLayers, optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001))
epochs = 25
model, history, results = trainAndEvaluate(model, epochs, train, val, test)
print(f"Training: {results}")
stats = getStats(history)

predTest(levels, model, results, "Aural/Datasets/Spectrogram Testing Datasets/", "Aural/Models/Saved/Results.csv", imgw, imgh, "StochasticSpectrogramsModelComplete", 2)

showStats(epochs, stats)
model.save("Aural/Models/Saved/StochasticSpectrogramImproved.h5")