# Experiment with simple convolutional neural network
# Visualize the filters the network learns

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds

# PART 1
## 1.1 Data loading and preprocessing
# Load mnist dataset 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

fig, ax = plt.subplots(2, 2, figsize=(5, 5))
ax[0,0].imshow(train_images[0], cmap="binary")
ax[0,0].axis('off')
ax[0,0].title.set_text(train_labels[0])
ax[0,1].imshow(train_images[1], cmap="binary")
ax[0,1].axis('off')
ax[0,1].title.set_text(train_labels[1])
ax[1,0].imshow(train_images[2], cmap="binary")
ax[1,0].axis('off')
ax[1,0].title.set_text(train_labels[2])
ax[1,1].imshow(train_images[3], cmap="binary")
ax[1,1].axis('off')
ax[1,1].title.set_text(train_labels[3])
plt.show()

# Normalize images
def normalize_img(image, label):
  return tf.cast(image, tf.float32) / 127 - 1, label

# Training pipeline
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# Evaluation pipeline
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

## 1.2 Network design
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(50, 5, padding = "valid", activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
    tf.keras.layers.Conv2D(50, 5, padding = "valid", activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.legacy.SGD(),
    metrics=['accuracy']
)

mod = model.fit(ds_train, batch_size=100, epochs=5, verbose=True) #Revert epochs to 200 later

# Plot of Accuracy
plt.plot(mod.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy across Training Period')
plt.show()

# Plot of Error
plt.plot(mod.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error across Training Period')
plt.show()

# PART 2
# Filter matrices
filter = np.transpose(model.layers[0].weights[0], (2,3,0,1))[0]

fig, ax = plt.subplots(10, 5)
fig.suptitle("First Layer Filters")
for filter, ax in zip(filter, ax.ravel()):
 ax.imshow(filter)
 ax.get_yaxis().set_visible(False)
 ax.get_xaxis().set_visible(False)
plt.show()

# What do they look like

# Scatter plot of embedding vectors (layer 7)
embed = model.layers[6].weights[0]
embed = np.transpose(embed)

embx = embed[0,:]
emby = embed[1,:]

fig, ax = plt.subplots()
scatter = ax.scatter(embx, emby) # add embed layers
fig.suptitle("Network Embedding")
plt.show()

# Rotate each digit 15 degrees fom 0 to 180 (13 sets by 10 digits)
# first create set of test images 
rot_index = []
rotatelabels = []
rotatedsamples = []

# Find the index of the first occurrence of each label in test_labels
for i in range(10):
    rot_index.append(np.where(test_labels == i)[0][0])
    rotatelabels.append(test_labels[rot_index[i]])
    rotatedsamples.append(test_images[rot_index[i]])

rotAccuracy = []
predictions = []
for i in range(13):
    storedRot = []
    for k in range(len(rotatedsamples)):
       storedRot.append(scipy.ndimage.rotate(test_images[k], 15*i, reshape = False))

    storedRot = np.array(storedRot)

    predictions.append(model.predict(storedRot, verbose=1))
    accuracy = np.mean(np.argmax(predictions, axis=1) == rotatelabels)
    rotAccuracy.append(accuracy)

# Check results
for k, result in enumerate(rotAccuracy):
   print(f"Rot. Angle: {15*k} degrees, Accuracy: {result}")

rotAngle = [15*k for k in range(len(rotAccuracy))]

plt.figure(rotAngle, rotAccuracy)
plt.title("Rotation Angle Accuracy")
plt.xlabel("Rotation (Degrees)")
plt.ylabel("Accuracy")
plt.show