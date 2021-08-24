import tensorflow as tf
import numpy as np
import os

# This CNN takes samples of two sizes extracted from a superpixel 
# segmentation and uses to help in its classification
tf.keras.backend.clear_session()

# accessing and loading dataset
root_path = '/Users/rafael/Documents/Capstone project/Final report stuff/Datasets/55x55 & 111x111'
os.chdir(root_path)

# %% loading dataset files
print('Loading dataset...')

# loading training set
dataset = np.load('55 & 111 training dataset.npz')

# loading samples, full images as side-info and labels
small_samples = dataset['samples_1']
large_samples = dataset['samples_2']
labels = dataset['labels']

# loading validation set
dataset = np.load('55 & 111 validation dataset.npz')

# loading samples, full images as side-info and labels
val_small_samples = dataset['samples_1']
val_large_samples = dataset['samples_2']
val_labels = dataset['labels']

# preprocessing datasets
small_samples /= 255.0
val_small_samples /= 255.0
large_samples /= 255.0
val_large_samples /= 255.0

val_data = ([val_small_samples, val_large_samples], val_labels)

# %% defining model structure
print('Defining model structure...')

tf.keras.backend.clear_session
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, concatenate

# inputs definition
small_sample_input = tf.keras.Input(shape = (55, 55, 3), name = 'small_samples')
large_sample_input = tf.keras.Input(shape = (111, 111, 3), name = 'large_samples')

# small samples branch
# stage 1
x1 = Conv2D(48, (5,5), activation = 'relu')(small_sample_input)
x1 = MaxPooling2D((3,3), (2,2))(x1)

# stage 2
x1 = Conv2D(64, (5,5), activation = 'relu')(x1)
x1 = MaxPooling2D((3,3), (2,2))(x1)

# stage 3
x1 = Conv2D(128, (3,3), activation = 'relu')(x1)

# stage 4
x1 = Conv2D(256, (3,3), activation = 'relu')(x1)
x1 = MaxPooling2D((3,3), (2,2))(x1)
x1 = Flatten()(x1)

# large samples branch
# stage 1
x2 = Conv2D(48, (5,5), activation = 'relu')(large_sample_input)
x2 = MaxPooling2D((3,3), (2,2))(x2)

# stage 2
x2 = Conv2D(64, (5,5), activation = 'relu')(x2)
x2 = MaxPooling2D((3,3), (2,2))(x2)

# stage 3
x2 = Conv2D(128, (5,5), activation = 'relu')(x2)

# stage 4
x2 = Conv2D(256, (3,3), activation = 'relu')(x2)
x2 = MaxPooling2D((3,3), (2,2))(x2)
x2 = Flatten()(x2)

# concatenated branches
x = concatenate([x1, x2])
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation = 'relu')(x)
x = Dense(64, activation = 'relu')(x)
x = Dropout(0.2)(x)

out = Dense(1, activation = 'sigmoid', name = 'output')(x)

model = tf.keras.Model(
    inputs = [small_sample_input, large_sample_input],
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
        if(logs.get('val_accuracy') > 0.913):
            print('\nReached desired accuracy. Stopping training.')
            self.model.stop_training = True

# training
model.fit(
    {'small_samples': small_samples, 'large_samples': large_samples},
    {'output': labels}, validation_data = val_data,
    epochs = 10, batch_size = 32, callbacks = [accuracyCallback()])

# %% saving the model as a SavedModel
print('Saving the model')
model.save('/Users/rafael/Documents/Capstone project/Final report stuff/Model weights/55x55 & 111x111')