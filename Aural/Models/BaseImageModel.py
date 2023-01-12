import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pathlib
import PIL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def loadDatasets(path, batchsize, imgh, imgw, val_split):
    data_dir = pathlib.Path(path)

    batch_size = batchsize
    img_height = imgh
    img_width = imgw

    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=val_split,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=val_split,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    class_names = train_ds.class_names
    print(f"Classes: {class_names}")
    print(train_ds)

    for image_batch, labels_batch in train_ds.take(1):
        print(f"feature shape: {image_batch.shape}")
        print(f"Labels shape: {labels_batch.shape}")
        break
    
    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    num_classes = len(class_names)

    return train_ds, val_ds, test_ds, num_classes

def loadPredDs(path, batchsize, imgh, imgw,):
    data_dir = pathlib.Path(path)

    batch_size = batchsize
    img_height = imgh
    img_width = imgw

    pred_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    class_names = pred_ds.class_names
    print(f"Classes: {class_names}")
    print(pred_ds)

    return pred_ds

def dataAugmentLayer(img_height, img_width):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(img_height,
                                        img_width,
                                        3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    return data_augmentation


def buildModel(
                img_height, 
                img_width, 
                num_classes, 
                hiddenLayers=[
                        layers.Conv2D(16, 3, padding='same', activation='relu'),
                        layers.MaxPooling2D(),
                        layers.Conv2D(32, 3, padding='same', activation='relu'),
                        layers.MaxPooling2D(),
                        layers.Conv2D(64, 3, padding='same', activation='relu'),
                        layers.MaxPooling2D(),
                        layers.Dropout(0.2),
                        layers.Flatten(),
                        layers.Dense(128, activation='relu')
                    ], 
                optimizer="adam", 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics=['accuracy'], 
                data_augmentation=False):
    model = Sequential()
    model.add(layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)))

    if data_augmentation:
        model.add(dataAugmentLayer(img_height, img_width))
    

    for layer in hiddenLayers:
        model.add(layer)
    
    model.add(layers.Dense(num_classes))

    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)

    model.summary()
    return model

def trainAndEvaluate(model, epochs, train_ds, val_ds, test_ds):
    epochs=epochs
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    results = model.evaluate(test_ds, return_dict=True)

    return model, history, results

def getStats(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    return [acc, val_acc, loss, val_loss]

def showStats(epochs, stats):
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, stats[0], label='Training Accuracy')
    plt.plot(epochs_range, stats[1], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, stats[2], label='Training Loss')
    plt.plot(epochs_range, stats[3], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def predEval(dataset, model):
    return [model.predict(dataset), model.evaluate(dataset, return_dict=True)]

def predTest(stochasLevels, model, normal_result, base_path, save_path, imgw, imgh, name="BaseModel", batch_size=2):
    try:
        results = pd.read_csv(save_path).to_dict("list")
        print(results)
    except:
        results = {"Model":[], "Normal_Loss": [], "Normal_Acc": []}
    results["Model"].append(name)
    results["Normal_Loss"].append(round(normal_result["loss"], 6))
    results["Normal_Acc"].append(round(normal_result["accuracy"], 6))
    for level in stochasLevels:
        pred_ds = loadPredDs(f"{base_path}{level}/", 32, imgh, imgw)
        base_result = predEval(pred_ds, model)[1]
        try:
            results[f"{level}_Loss"].append(round(base_result['loss'], 6))
            results[f"{level}_Acc"].append(round(base_result['accuracy'], 6))
        except:
            results[f"{level}_Loss"] = []
            results[f"{level}_Acc"] = []
            results[f"{level}_Loss"].append(round(base_result['loss'], 6))
            results[f"{level}_Acc"].append(round(base_result['accuracy'], 6))

        print(f"{level}: {results[f'{level}_Loss']}")
        print(f"{level}: {results[f'{level}_Acc']}")

    df = pd.DataFrame(results)
    df = df.reset_index()
    del df['index']
    df.to_csv(save_path, index=False)

    return results


if __name__ == "__main__":
    imgh, imgw = 180, 180
    train, val, test, numclasses = loadDatasets("Visual/Normal Datasets/Flowers", 32, imgh, imgw, 0.2)
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

    model = buildModel(imgh, imgw, numclasses, data_augmentation=True, hiddenLayers=hiddenLayers)
    epochs = 150
    model, history, results = trainAndEvaluate(model, epochs, train, val, test)
    print(results)
    stats = getStats(history)
    showStats(epochs, stats)

    model.save("/home/paradox/Documents/VSCODE Projects/BTYSTE-2023/Visual/Models/Saved/Flower.h5")


