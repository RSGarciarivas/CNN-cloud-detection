import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.utils import plot_model

# This CNN takes 55x55 pixel samples extracted from a superpixel segmentation and uses
# for its classification
tf.keras.backend.clear_session()

# accessing and loading dataset
root_path = '/Users/rafael/Documents/Capstone project/Final report stuff/Datasets/55x55 & 111x111'
os.chdir(root_path)

# %% loading dataset files
print('Loading dataset...')

# loading training set
dataset = np.load('55 & 111 training dataset.npz')

# loading samples, full images as side-info and labels
samples = dataset['samples_1']
labels = dataset['labels']

# loading validation set
dataset = np.load('55 & 111 validation dataset.npz')

# loading samples, full images as side-info and labels
val_samples = dataset['samples_1']
val_labels = dataset['labels']

# preprocessing datasets
samples /= 255.0
val_samples /= 255.0

val_data = (val_samples, val_labels)

# %% defining model structure
print('Defining model structure...')

tf.keras.backend.clear_session
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# inputs definition
sample_input = tf.keras.Input(shape = (55, 55, 3), name = 'samples')

# stage 1
x = Conv2D(48, (5,5), activation = 'relu')(sample_input)
x = MaxPooling2D((3,3), (2,2))(x)

# stage 2
x = Conv2D(64, (5,5), activation = 'relu')(x)
x = MaxPooling2D((3,3), (2,2))(x)

# stage 3
x = Conv2D(128, (3,3), activation = 'relu')(x)

# stage 4
x = Conv2D(256, (3,3), activation = 'relu')(x)
x = MaxPooling2D((3,3), (2,2))(x)
x = Flatten()(x)

# fully-connected branches
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation = 'relu')(x)
x = Dense(64, activation = 'relu')(x)
x = Dropout(0.2)(x)

out = Dense(1, activation = 'sigmoid', name = 'output')(x)

model = tf.keras.Model(
    inputs = sample_input,
    outputs = out)

# model.summary()
plot_model(model,
           to_file = '55x55 CNN model.png',
           show_shapes = False,
           show_dtype = False,
           show_layer_names = True,
           rankdir = 'TB',
           expand_nested = False,
           dpi = 96
)

# %% compiling and training the model
print('Compiling and training the model...')

# compiling
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy', 'Precision', 'Recall'])

# training
model.fit(
    {'samples': samples},
    {'output': labels}, validation_data = val_data,
    epochs = 10, batch_size = 32)

# %% saving the model as a SavedModel
print('Saving the model')
model.save('/Users/rafael/Documents/Capstone project/Final report stuff/Model weights/55x55')