import pandas as pd
import numpy as np
import time
import pathlib

import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""### Visualize Function"""

def acc_loss_graph(history):
  #-----------------------------------------------------------
  # Retrieve a list of list results on training and test data
  # sets for each training epoch
  #-----------------------------------------------------------
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(acc))
  #------------------------------------------------
  # Plot training and validation accuracy per epoch
  #------------------------------------------------
  plt.plot(epochs, acc, 'b', label='Training accuracy')
  plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
  plt.legend()
  plt.title('Training and validation accuracy')

  plt.figure()
  #------------------------------------------------
  # Plot training and validation loss per epoch
  #------------------------------------------------
  plt.plot(epochs, loss, 'b', label='Training Loss')
  plt.plot(epochs, val_loss, 'r', label='Validation Loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.show()

"""### className"""

classNames = {0: 'Speed limit (20km/h)',
 1: 'Speed limit (30km/h)',
 2: 'Speed limit (50km/h)',
 3: 'Speed limit (60km/h)',
 4: 'Speed limit (70km/h)',
 5: 'Speed limit (80km/h)',
 6: 'End of speed limit (80km/h)',
 7: 'Speed limit (100km/h)',
 8: 'Speed limit (120km/h)',
 9: 'No passing',
 10: 'No passing for vehicles over 3.5 metric tons',
 11: 'Right-of-way at the next intersection',
 12: 'Priority road',
 13: 'Yield',
 14: 'Stop',
 15: 'No vehicles',
 16: 'Vehicles over 3.5 metric tons prohibited',
 17: 'No entry',
 18: 'General caution',
 19: 'Dangerous curve to the left',
 20: 'Dangerous curve to the right',
 21: 'Double curve',
 22: 'Bumpy road',
 23: 'Slippery road',
 24: 'Road narrows on the right',
 25: 'Road work',
 26: 'Traffic signals',
 27: 'Pedestrians',
 28: 'Children crossing',
 29: 'Bicycles crossing',
 30: 'Beware of ice/snow',
 31: 'Wild animals crossing',
 32: 'End of all speed and passing limits',
 33: 'Turn right ahead',
 34: 'Turn left ahead',
 35: 'Ahead only',
 36: 'Go straight or right',
 37: 'Go straight or left',
 38: 'Keep right',
 39: 'Keep left',
 40: 'Roundabout mandatory',
 41: 'End of no passing',
 42: 'End of no passing by vehicles over 3.5 metric tons'}

"""### Connect to GD and unzip """

from google.colab import drive
drive.mount('/content/gdrive')

#!unzip -q /content/gdrive/MyDrive/dataset.zip -d .

"""### shuffle (optional)

"""

# import random
# all_image_paths1= np.array(all_image_paths)
# shuffle_index= np.arange(len(all_image_paths1))
# random.shuffle(shuffle_index)
# shuffle_index
# all_image_paths1= all_image_paths1[shuffle_index]
# all_labels1= all_labels1[shuffle_index]
# all_labels1= np.array(all_labels)
# all_labels1.shape

"""## Work

"""

import os
os.getcwd()

num_cate= len(os.listdir('/content/test'))
num_cate

"""----> all path train"""

data_dir = '/content/Train'
data_dir = pathlib.Path(data_dir)

all_image_paths = list(data_dir.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
all_image_paths[0]

"""----> all path test"""

data_dir_test = '/content/Test'
data_dir_test = pathlib.Path(data_dir_test)

all_image_paths_test = list(data_dir_test.glob('*'))
all_image_paths_test = [str(path) for path in all_image_paths_test]
all_image_paths_test[0]

"""-----> all labels"""

all_labels = list(map(lambda x: x.split('/')[-2], all_image_paths))
all_labels[0:5]

"""----> split, get Train, Validation set"""

from sklearn.model_selection import train_test_split 
x_train, x_val, y_train, y_val = train_test_split(all_image_paths, all_labels, test_size=0.2)

"""---> get Test"""

test = pd.read_csv('/content/Test.csv')

y_test = test["ClassId"].values
x_test = '/content/' + test["Path"].values

"""Test set for predict"""

from PIL import Image
data=[]
for img in x_test:
    image = Image.open(img)
    image = image.resize((32,32))
    data.append(np.array(image))

X_test= np.array(data)
X_test= X_test/255.0
labels= test["ClassId"].values

"""Number of data in each set"""

print(len(x_train))
print(len(x_val))
print(len(X_test))

"""### Plot classes"""

key, value= np.unique(all_labels,return_counts=True)

np.unique(all_labels,return_counts=True)

plt.figure(figsize=(15,5))
plt.bar(key,value)

"""### Pre-processing"""

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [32, 32])
    image /= 255.0    # Normalize to [0,1] range
    return image

def load_and_preprocess_image(path, label):
    image_raw = tf.io.read_file(path)
    image = preprocess_image(image_raw)
    return image, label

"""----> one-hot"""

from tensorflow import keras
y_train = keras.utils.to_categorical(y_train, 43)
y_val = keras.utils.to_categorical(y_val, 43)
y_test = keras.utils.to_categorical(y_test, 43)

"""----> to Dataset"""

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test))

"""Batch 32"""

train_ds_32 = train_ds.map(load_and_preprocess_image).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_32 = val_ds.map(load_and_preprocess_image).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_32 = test_ds.map(load_and_preprocess_image).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

"""Batch 64"""

train_ds_64 = train_ds.map(load_and_preprocess_image).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_64 = val_ds.map(load_and_preprocess_image).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_64 = test_ds.map(load_and_preprocess_image).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

"""### Visualization"""

images, label = next(iter(train_ds_32.take(1)))
plt.imshow(images[1])

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

for images, label in train_ds_32.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])   
    plt.axis("off")

"""##CNN """

from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import layers

Cnn = tf.keras.Sequential([
      layers.Conv2D(16, (3,3), activation='relu', input_shape=(32,32,3), padding='same'),
      BatchNormalization(),
      layers.Conv2D(16, (3,3), activation='relu', padding='same'),
      BatchNormalization(),
      layers.MaxPooling2D(pool_size=(2,2)),
      layers.Dropout(0.2),
      layers.Conv2D(32, (3,3), activation='relu', padding='same'),
      BatchNormalization(),
      layers.Conv2D(32, (3,3), activation='relu', padding='same'),
      BatchNormalization(),
      layers.MaxPooling2D(pool_size=(2,2)),
      layers.Dropout(0.2),
      layers.Conv2D(64, (3,3), activation='relu', padding='same'),
      BatchNormalization(),
      layers.Conv2D(64, (3,3), activation='relu', padding='same'),
      BatchNormalization(),
      layers.MaxPooling2D(pool_size=(2,2)),
      layers.Dropout(0.2),
      layers.Flatten(),
      Dense(2048, activation='relu'),
      layers.Dropout(0.3),
      Dense(1024, activation='relu'),
      layers.Dropout(0.3),
      Dense(128, activation='relu'),
      layers.Dropout(0.3),
      Dense(43, activation='softmax')
    ])

from tensorflow.keras.optimizers import Adam
Cnn.compile(loss='categorical_crossentropy', 
            optimizer=Adam(learning_rate=1e-4), 
            metrics=['accuracy'])

"""###Batch 32"""

epochs=30
history = Cnn.fit(train_ds_32,               
                  validation_data=val_ds_32, 
                  epochs=epochs)

"""Cnn Evaluate

30 epoch
"""

acc_loss_graph(history)

Cnn.evaluate(val_ds_32)

Cnn.evaluate(test_ds_32)

Cnn.save('/content/gdrive/MyDrive/Colab Notebooks/Cnn32_traffic_classifier_test.h5')

from keras.models import load_model
Cnn_32 = load_model('/content/gdrive/MyDrive/Colab Notebooks/Cnn32_traffic_classifier_test.h5')

pred_32= Cnn.predict(X_test)
pred_indices_32 = np.argmax(pred_32,axis=1)

from sklearn.metrics import classification_report
print(classification_report(labels, pred_indices_32))

import seaborn as sns
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(labels, pred_indices_32)

df_cm = pd.DataFrame(cf, index = classNames,  columns = classNames)
plt.figure(figsize = (15,15))
sns.heatmap(df_cm, annot=True)

"""####Predict on image"""

from google.colab import files

# UPLOAD A PHOTO and PREDICT
uploaded = files.upload()
for fn in uploaded.keys():
  print(fn)
  img_path = './' + fn

from keras.preprocessing import image
import matplotlib.image as mpimg

img        = image.load_img(img_path, target_size=(32, 32))
img_array  = image.img_to_array(img)
img_array  = np.expand_dims(img_array, axis=0)

prediction = Cnn.predict(img_array)
# lay vi tri class co pro max
value = prediction[0].argmax()
pred = classNames[value]

plt.figure(figsize=(5,5))
img = mpimg.imread(img_path)
plt.imshow(img)
plt.title('Prediction: ' + pred.upper())
plt.axis('off')
plt.grid(b=None)
plt.show()

"""###Batch 64"""

epochs=30
history = Cnn.fit(train_ds_64,               
                  validation_data=val_ds_64, 
                  epochs=epochs)

acc_loss_graph(history)

Cnn.evaluate(val_ds_64)

Cnn.evaluate(test_ds_64)

Cnn.save('/content/gdrive/MyDrive/Colab Notebooks/Cnn64_traffic_classifier_test.h5')

pred_64 = Cnn.predict(X_test)
pred_indices_64 = np.argmax(pred_64,axis=1)

"""#### Evaluate batch64"""

from sklearn.metrics import classification_report
print(classification_report(labels, pred_indices_64))

import seaborn as sns
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(labels, pred_indices)

df_cm = pd.DataFrame(cf, index = classNames,  columns = classNames)
plt.figure(figsize = (15,15))
sns.heatmap(df_cm, annot=True)

