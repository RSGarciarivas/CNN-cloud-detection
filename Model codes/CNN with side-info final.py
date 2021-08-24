import tensorflow as tf
import numpy as np
import os

# This CNN takes samples extracted from a superpixel segmentation and uses
# the whole picture as side information to help in its classification
tf.keras.backend.clear_session()

# accessing and loading dataset
root_path = '/Users/rafael/Documents/Capstone project/Final report stuff/Datasets/75x75 side-info'
os.chdir(root_path)

# %% loading dataset files
print('Loading dataset...')

# loading training set
dataset = np.load('75x75 side-info training dataset.npz')

# loading samples, full images as side-info and labels
samples = dataset['samples']
side_info = dataset['side_info']
labels = dataset['labels']

# loading validation set
dataset = np.load('75x75 side-info validation dataset.npz')

# loading samples, full images as side-info and labels
val_samples = dataset['samples']
val_side_info = dataset['side_info']
val_labels = dataset['labels']

# preprocessing datasets
samples /= 255.0
val_samples /= 255.0
side_info /= 255.0
val_side_info /= 255.0

val_data = ([val_samples, val_side_info], val_labels)

# %% defining model structure
print('Defining model structure...')

tf.keras.backend.clear_session
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, concatenate

# inputs definition
sample_input = tf.keras.Input(shape = (75, 75, 3), name = 'samples')
side_info_input = tf.keras.Input(shape = (90, 160, 3), name = 'side-info')

# sample branch
# stage 1
x1 = Conv2D(32, (3,3), activation = 'relu')(sample_input)
x1 = MaxPooling2D((2,2))(x1)

# stage 2
x1 = Conv2D(64, (3,3), activation = 'relu')(x1)
x1 = MaxPooling2D((2,2))(x1)

# stage 3
x1 = Conv2D(64, (3,3), activation = 'relu')(x1)
x1 = MaxPooling2D((2,2))(x1)
x1 = Flatten()(x1)

# side-info branch
# stage 1
x2 = Conv2D(32, (3,3), activation = 'relu')(side_info_input)
x2 = MaxPooling2D((2,3))(x2)

# stage 2
x2 = Conv2D(64, (3,3), activation = 'relu')(x2)
x2 = MaxPooling2D((2,3))(x2)

# stage 3
x2 = Conv2D(64, (3,3), activation = 'relu')(x2)
x2 = MaxPooling2D((2,2))(x2)
x2 = Flatten()(x2)

# concatenated branches
x = concatenate([x1, x2])
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation = 'relu')(x)
x = Dropout(0.2)(x)

out = Dense(1, activation = 'sigmoid', name = 'output')(x)

model = tf.keras.Model(
    inputs = [sample_input, side_info_input],
    outputs = out)

model.summary()

# %% compiling and training the model
print('Compiling and training the model...')

# compiling
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy', 'Precision', 'Recall'])

# callback for stopping when given accuracy is reached
class accuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('val_accuracy') > 0.917):
            print('\nReached desired accuracy. Stopping training.')
            self.model.stop_training = True

# training
model.fit(
    {'samples': samples, 'side-info': side_info},
    {'output': labels}, validation_data = val_data,
    epochs = 10, batch_size = 32, callbacks = [accuracyCallback()])

# %% saving the model as a SavedModel
print('Saving the model')
model.save('/Users/rafael/Documents/Capstone project/Final report stuff/Model weights/75x75 side-info')