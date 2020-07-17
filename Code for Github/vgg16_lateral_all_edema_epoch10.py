import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys



HEIGHT = 224
WIDTH = 224

df = pd.read_csv("/home/jyarnal1/lateral_all_edema.csv", header=0, sep=',')




train_df = df.sample(frac=0.7, random_state=1)
tmp_df = df.drop(train_df.index)
test_df = tmp_df.sample(frac=0.33333, random_state=1)
valid_df = tmp_df.drop(test_df.index)




#function to get list of images from dataframe
def get_image_list(df):
    image_list = []
    for i in range(len(df)):
        image_list.append(df.iloc[i,0])
    return image_list

#function to get list of labels from dataframe
def get_label_list(df):
    labels_list = []
    for i in range(len(df)):
        labels_list.append(df.iloc[i,1:].to_numpy())
    return labels_list

#function to process image files
def process_path(file):
    image = tf.io.read_file(file)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = tf.image.grayscale_to_rgb(image, name=None)
    image /= 255.0  # normalize to [0,1] range
    #print(image)
    return image#, label

#get training tensor
train_image_dataset = tf.data.Dataset.from_tensor_slices(get_image_list(train_df))
train_labels_dataset = tf.data.Dataset.from_tensor_slices(get_label_list(train_df))
train_image_ds = train_image_dataset.map(process_path)
train_dataset = tf.data.Dataset.zip((train_image_ds, train_labels_dataset))

#get testing tensor
test_image_dataset = tf.data.Dataset.from_tensor_slices(get_image_list(test_df))
test_labels_dataset = tf.data.Dataset.from_tensor_slices(get_label_list(test_df))
test_image_ds = test_image_dataset.map(process_path)
test_dataset = tf.data.Dataset.zip((test_image_ds, test_labels_dataset))

#get validation tensor
valid_image_dataset = tf.data.Dataset.from_tensor_slices(get_image_list(valid_df))
valid_labels_dataset = tf.data.Dataset.from_tensor_slices(get_label_list(valid_df))
valid_image_ds = valid_image_dataset.map(process_path)
valid_dataset = tf.data.Dataset.zip((valid_image_ds, valid_labels_dataset))

#from here https://www.tensorflow.org/tutorials/images/transfer_learning

IMG_SHAPE = (HEIGHT, WIDTH, 3)


base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 500

train_batches = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = valid_dataset.batch(BATCH_SIZE)
test_batches = test_dataset.batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
    pass

image_batch.shape

feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

# Let's take a look at the base model architecture
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1, activation = 'relu')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.summary()

initial_epochs = 10
validation_steps=20

#use afterwards with test data
loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

#use afterwards with test data
losst,accuracyt = model.evaluate(test_batches)

with open("vgg16_lateral_all_edema_epoch10_accuracy_v2.txt", "w") as output:
    output.write(str(accuracyt))

history = history.history

with open("vgg16_lateral_all_edema_epoch10_history_v2.txt", "w") as output:
    output.write(str(history))

predictions = model.predict(test_batches)
predictions = np.round(predictions)
predictions = tf.keras.backend.flatten(
    predictions
)

labels = np.asarray(get_label_list(test_df)).astype(np.float32)
labels = tf.keras.backend.flatten(
    labels
)

cm = tf.math.confusion_matrix(labels=labels, predictions=predictions)

print(cm)

with open("vgg16_lateral_all_edema_epoch10_cm_v2.txt", "w") as output:
    output.write(str(cm))
