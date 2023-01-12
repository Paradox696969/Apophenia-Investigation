import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers


def loadDataset(path, remove_func=False, keep_func=False):
    df = pd.read_csv(path)

    features = df.copy()
    labels = features.pop("Last")
    if remove_func and not keep_func:
        features.pop("func")
    elif keep_func:
        labels = features.pop("func")
    
    return np.array(features), labels

features, labels = loadDataset("Numerical/Datasets/CSV/Stochas.csv", False, True)


model = tf.keras.Sequential(
    [
        layers.Dense(32, activation="elu", kernel_regularizer=tf.keras.regularizers.L2(0.01)),
        layers.Dense(64, activation="elu", kernel_regularizer=tf.keras.regularizers.L2(0.01)),
        layers.Dense(6)
    ]
)

model.compile(  
                loss = tf.keras.losses.MeanSquaredError(),
                optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01),
                metrics=["accuracy"]
            )

model.fit(features, labels, epochs=10, validation_split=0.2, batch_size=128)

def save(name):
    result_str = f"{name},"
    test_features, test_labels = loadDataset("Numerical/Datasets/CSV/LowTest.csv", False, True)
    result = model.evaluate(test_features, test_labels)[0]
    result_str += f"{round(result, 6)},"
    test_features, test_labels = loadDataset("Numerical/Datasets/CSV/StochasTest.csv", False, True)
    result = model.evaluate(test_features, test_labels)[0]
    result_str += f"{round(result, 6)}"

    with open("Numerical/Models/CSVModels/Results.csv", "a") as f:
        f.write(f"\n{result_str}")

save("Stochaspattern")

model.save("Numerical/Models/Saved/Pattern-StochasCSV_Model.h5")