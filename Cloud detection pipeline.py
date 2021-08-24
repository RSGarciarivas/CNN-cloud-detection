import tensorflow as tf
import numpy as np
import cv2 as cv
import os

from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
# %% Load selected model
tf.keras.backend.clear_session

saved_model_path = '/Users/rafael/Documents/Capstone project/Final report stuff/Model weights/'
model_name = '75x75 side-info'

model = tf.keras.models.load_model(saved_model_path + model_name)

# %% Load image and segment it into superpixels

image_path = '/Users/rafael/Documents/Capstone project/Datasets/HRC_WHU by train & test/Images/Training/'
image_name = 'water_10'

image = cv.imread(image_path + image_name + '.tif')

# read reference mask and process it (1 = cloud pixel, 0 = non-cloud)
reference_mask_image = cv.imread(image_path + image_name + '_ReferenceMask.tif')
reference_mask = reference_mask_image[:,:,0] // 255

# SLIC segmentation parameters
compactness = 35
approx_n_segments = 200

if(model_name == '55x55'):
    sample_size_1 = 55
    half_length_1 = (sample_size_1 - 1)/2
    sample_requisite = half_length_1
    
elif(model_name == '55x55 & 111x111'):
    sample_size_1 = 55
    sample_size_2 = 111
    half_length_1 = (sample_size_1 - 1)/2           # used for extracting samples
    half_length_2 = (sample_size_2 - 1)/2
    sample_requisite = half_length_2
    
elif(model_name == '75x75 transfer learning'):
    sample_size_1 = 75
    half_length_1 = (sample_size_1 - 1)/2
    sample_requisite = half_length_1
    
elif(model_name == '75x75 side-info'):
    sample_size_1 = 75
    half_length_1 = (sample_size_1 - 1)/2
    H = image.shape[0] // 8
    W = image.shape[1] // 8
    image_downsized = cv.resize(image, (W, H))
    sample_requisite = half_length_1
    
# x and y ranges from which to extract samples
x_range = image.shape[1] - 1
y_range = image.shape[0] - 1
    
# SLIC segmentation into superpixels
slic_labels = slic(image, n_segments = approx_n_segments,
                   compactness = compactness, sigma = 1, start_label = 1)

# get image with boundaries
marked_image = mark_boundaries(image, slic_labels, color = (255, 0, 0))

# use regionprops to obtain superpixel centroids
regions = regionprops(slic_labels)
n_labels_raw = len(regions)                 # resulting number of superpixels
    
# arrays for storing centroids (ignoring boundary limits)
cy_raw = np.zeros(n_labels_raw)
cx_raw = np.zeros(n_labels_raw)

# extract centroid for each superpixel
for i in range(n_labels_raw):
    cy_raw[i], cx_raw[i] = regions[i].centroid

# only use centroids that fit the sample size
cy = cy_raw[(cy_raw > sample_requisite) & (y_range - cy_raw > sample_requisite)
            & (cx_raw > sample_requisite) & (x_range - cx_raw > sample_requisite)]
    
cx = cx_raw[(cy_raw > sample_requisite) & (y_range - cy_raw > sample_requisite)
            & (cx_raw > sample_requisite) & (x_range - cx_raw > sample_requisite)]     
    
n_labels = len(cx)

# arrays for storing samples
samples_1 = np.zeros((n_labels, sample_size_1, sample_size_1, 3))

if(model_name == '55x55 & 111x111'):
    samples_2 = np.zeros((n_labels, sample_size_2, sample_size_2, 3))
    samples_2_downsized = np.zeros((n_labels, sample_size_1, sample_size_1, 3))
    
elif(model_name == '75x75 side-info'):
    image_downsized_samples = np.zeros((n_labels, image_downsized.shape[0], image_downsized.shape[1], 3))
    
# array for storing assigned labels
labels = np.zeros(n_labels)
labels = labels.astype(int)

# %% Extract samples

for j in range(len(cy)):
    
    # round to closest pixel
    x = np.round(cx[j])
    y = np.round(cy[j])
        
    # x range for extracting sample 1
    x_init_1 = np.intc(x - half_length_1)
    x_end_1 = np.intc(x + half_length_1 + 1)
    # y range for extracting sample 1
    y_init_1 = np.intc(y - half_length_1)
    y_end_1 = np.intc(y + half_length_1 + 1)
        
    # extract RGB values of sample 1
    samples_1[j, :, :, :] = image[y_init_1 : y_end_1, x_init_1 : x_end_1, :]
        
    if(model_name == '55x55 & 111x111'):
        # x range for extracting sample 2
        x_init_2 = np.intc(x - half_length_2)
        x_end_2 = np.intc(x + half_length_2 + 1)
        # y range for extracting sample 2
        y_init_2 = np.intc(y - half_length_2)
        y_end_2 = np.intc(y + half_length_2 + 1)
            
        # extract RGB values for sample 2
        samples_2[j, :, :, :] = image[y_init_2 : y_end_2, x_init_2 : x_end_2, :]
        samples_2_downsized[j] = cv.resize(samples_2[j], (sample_size_1, sample_size_1))
    
    elif(model_name == '75x75 side-info'):
        # add downsized image for each sample
        image_downsized_samples[j] = image_downsized
    
    # label sample by comparing to its reference mask
    sample_mask = reference_mask[y_init_1 : y_end_1, x_init_1 : x_end_1]
        
    # obtain cloud pixels ratio within small sample and decide label
    cloud_score = np.sum(sample_mask, axis = None) / (sample_size_1 ** 2)
    if(cloud_score >= 0.5):
        labels[j] = 1

# %% Classification

# prepare samples to feed into the model
if(model_name == '55x55' or model_name == '75x75 transfer learning'):
    x = samples_1 / 255.0

elif(model_name == '55x55 & 111x111'):
    x = [samples_1 / 255.0, samples_2_downsized / 255.0]

elif(model_name == '75x75 side-info'):
    x = [samples_1 / 255.0, image_downsized_samples / 255.0]

# pass samples through model
predictions = model.predict(x)

# %% Display classification output

# obtain centroids of detected cloud superpixels
centroids = []
for i in range(n_labels):
    if(predictions[i] >= 0.5):
        centroids.append([cy[i].astype(int),cx[i].astype(int)])

# obtain slic labels of those centroids
superpixel_cloud_labels = []
for x, y in centroids:
    superpixel_cloud_labels.append(slic_labels[x, y])

slic_labels_1d = np.reshape(slic_labels, -1)
results_mask_1d = np.zeros(slic_labels_1d.shape)

for i in superpixel_cloud_labels:
    results_mask_1d[slic_labels_1d == i] = 1

results_mask = np.reshape(results_mask_1d, slic_labels.shape)

img_result = np.copy(image)

img_result[results_mask == 1] = (255, 255, 255)

# show images
cv.imshow('Cloud', image)
superpixels = np.copy(marked_image)
cv.imshow('Superpixel segmentation', superpixels)
cv.imshow('Output', img_result)

cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)