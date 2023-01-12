import collections
import pathlib

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text

batch_size = 32
seed = 42

raw_train_ds = utils.text_dataset_from_directory(
    "Textual/Datasets/Text",
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_val_ds = utils.text_dataset_from_directory(
    "Textual/Datasets/Text",
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

num_classes = len(raw_train_ds.class_names)
# print(raw_train_ds.class_names)

# for text_batch, label_batch in raw_train_ds.take(1):
#   for i in range(10):
#     print("String: ", text_batch.numpy()[i])
#     print("LastLetter:", label_batch.numpy()[i])

raw_test_ds = raw_val_ds.shard(num_shards=2, index=0)
raw_val_ds = raw_val_ds.shard(num_shards=2, index=1)

VOCAB_SIZE = 1000

binary_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='binary')

MAX_SEQUENCE_LENGTH = 100

int_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)

train_text = raw_train_ds.map(lambda text, labels: text)
binary_vectorize_layer.adapt(train_text)
int_vectorize_layer.adapt(train_text)

def binary_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return binary_vectorize_layer(text), label

text_batch, label_batch = next(iter(raw_train_ds))
first_string, first_label = text_batch[0], label_batch[0]
# print("String", first_string)
# print("LastLetter", first_label)

# print("'binary' vectorized string:",
#       binary_vectorize_text(first_string, first_label)[0])


binary_train_ds = raw_train_ds.map(binary_vectorize_text)
binary_val_ds = raw_val_ds.map(binary_vectorize_text)
binary_test_ds = raw_test_ds.map(binary_vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)

binary_train_ds = configure_dataset(binary_train_ds)
binary_val_ds = configure_dataset(binary_val_ds)
binary_test_ds = configure_dataset(binary_test_ds)

binary_model = tf.keras.Sequential([
    layers.Dense(128), 
    layers.Dense(64), 
    layers.Dense(num_classes)
    ])

binary_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

history = binary_model.fit(
    binary_train_ds, validation_data=binary_val_ds, epochs=20)

binary_model.save("Textual/Models/Saved/Unoptimised.h5")