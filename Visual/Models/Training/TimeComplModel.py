from BaseModel import *
import keras
import tensorflow_datasets as tfds

from BaseModel import *
import keras
import tensorflow_datasets as tfds
import time

def save(path, value):
    with open(path, "a") as f:
        f.write(value)


def normalModel(training, name, ntest, stest):
    keras.backend.clear_session()


    imgh, imgw = 100, 100
    epochs = 1
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

    train, val, test, numclasses = loadDatasets(training, 128, imgh, imgw, 0.2)

    model = buildModel(imgh, imgw, numclasses, data_augmentation=True, hiddenLayers=hiddenLayers)
    start_time = time.time()

    model, history, normal_result = trainAndEvaluate(model, epochs, train, val, test)

    nds = loadPredDs(ntest, 2, imgh, imgw)
    sds = loadPredDs(stest, 2, imgh, imgw)
    normalr = model.evaluate(nds)
    stochasr = model.evaluate(sds)
    print(normalr, stochasr)

    end_time = time.time()
    time_taken = (end_time - start_time) / 60
    losstn = normalr[0] / time_taken
    acctn = normalr[1] / time_taken
    lossts = stochasr[0] / time_taken
    accts = stochasr[1] / time_taken

    result_str = f"\n{name},{time_taken},{losstn},{acctn},{losstn},{acctn}"
    save("Visual/Models/Training/TimeComplResults/Results.csv", result_str)


def stochasticModel(training, name, ntest, stest):
    keras.backend.clear_session()


    imgh, imgw = 100, 100
    epochs = 1
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

    train, val, test, numclasses = loadDatasets(training, 128, imgh, imgw, 0.2)

    model = buildModel(imgh, imgw, numclasses, data_augmentation=True, hiddenLayers=hiddenLayers, optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001))
    start_time = time.time()

    model, history, normal_result = trainAndEvaluate(model, epochs, train, val, test)

    nds = loadPredDs(ntest, 2, imgh, imgw)
    sds = loadPredDs(stest, 2, imgh, imgw)
    normalr = model.evaluate(nds)
    stochasr = model.evaluate(sds)
    print(normalr, stochasr)

    end_time = time.time()
    time_taken = (end_time - start_time) / 60
    losstn = normalr[0] / time_taken
    acctn = normalr[1] / time_taken
    lossts = stochasr[0] / time_taken
    accts = stochasr[1] / time_taken

    result_str = f"\n{name},{time_taken},{losstn},{acctn},{losstn},{acctn}"
    save("Visual/Models/Training/TimeComplResults/Results.csv", result_str)

for folder in sorted(os.listdir("Visual/Stochastic Datasets/Merged TimeCL")):
    if folder.startswith("Normal"):
        ...
        # normalModel(f"Visual/Stochastic Datasets/Merged TimeCL/{folder}", folder, "Visual/Stochastic Datasets/Merged TimeCL TestN", "Visual/Stochastic Datasets/Merged TimeCL TestS")
    else:
        stochasticModel(f"Visual/Stochastic Datasets/Merged TimeCL/{folder}", folder, "Visual/Stochastic Datasets/Merged TimeCL TestN", "Visual/Stochastic Datasets/Merged TimeCL TestS")

    