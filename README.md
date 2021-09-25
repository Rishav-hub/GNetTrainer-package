# GNetTrainer - Application for training and preicting classification models.

![gnetimage](GNetTrainer/static/css/assets/img/GNet.png)

Official Repository for GNetTrainer

## Author

- [Rishav Dash](https://www.linkedin.com/in/rishdash/)
- Mail me at - 9930046@gmail.com
- Github - https://github.com/Rishav-hub/

## About GNetTrainer

GNetTrainer is a Deep Learning web application for training and predicting classification models which is written in Python 3. At the backend it uses Frameworks like Tensorflow 2.0 and Keras. You can train any classification model with any data using GNetTrainer without writing any line of code.

## Motivation

The main aim is to make something like Keras which is a high-level neural network library that runs on top of TensorFlow.

## How Image Classification looks using Keras

Import the required libraries

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
```
Define Paths
```python
ROOT = 'H:\\Parsonal\\Coding Practice\\dogCat'
os.chdir(ROOT)
os.getcwd()
```

Apply Augmentation
```python
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)


TRAIN_DIR = '/content/flowers_filtered/train'
TEST_DIR = '/content/flowers_filtered/val'

training_set = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical'
                                                 )

test_set = test_datagen.flow_from_directory(TEST_DIR, 
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical'
                                                 )
                                        
```
Download the pretrained model

```python
model = DenseNet169(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
```

Freeze layers
```python   
from keras.layers import BatchNormalization
for layer in model_base.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False
```

Add Custom Layers

```python
from keras.layers import BatchNormalization
for layer in model_base.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False
```
Adding Custom Layers

```python

model = Sequential()

model.add(tf.keras.layers.experimental.preprocessing.Resizing(224, 
                        224, interpolation="bilinear")) 

model.add(model_base)

model.add(Flatten())
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(64, activation='elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(5, activation='softmax'))
```

Defining Optimizers, Loss Functions and Checkpoints
```python
OPTIMIZERS = optimizers.Adam()

checkpoints = tf.keras.callbacks.ModelCheckpoint(
    'Densnet_model_best.hdf5',
    monitor="val_loss",
    verbose=0,
    save_best_only=False)


lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5) 

model.compile(optimizer= OPTIMIZERS, loss='categorical_crossentropy', metrics=['acc'])
```
Defining Path to Sabe the Model
```python
import time
import os

def saveModel_path(model_dir="/content/drive/MyDrive/DLCVNLP/Computer_Vision/SAVED_MODELS"):
    os.makedirs(model_dir, exist_ok=True)
    fileName = time.strftime("DensNetModel_%Y_%m_%d_%H_%M_%S_.h5")    
    model_path = os.path.join(model_dir, fileName)
    print(f"your model will be saved at the following location\n{model_path}")
    return model_path
```

Tensorboard Callback

```python
log_dir = get_log_path()
tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
```

Compiling Model
```python
model.compile(optimizer= OPTIMIZERS, loss='categorical_crossentropy', metrics=['acc'])
```
Fit Model and Start Training

```python
model.fit(training_set, 
          steps_per_epoch= 3452 // 32, 
          epochs = 10,
          validation_data = test_set,
          validation_steps = 10,
          callbacks= [checkpoints, lr_scheduler])
```
```python
Epoch 1/10
107/107 [==============================] - 112s 817ms/step - loss: 0.9201 - acc: 0.6760 - val_loss: 0.5511 - val_acc: 0.8313
Epoch 2/10
107/107 [==============================] - 80s 746ms/step - loss: 0.4234 - acc: 0.8681 - val_loss: 0.3199 - val_acc: 0.8906
Epoch 3/10
107/107 [==============================] - 80s 747ms/step - loss: 0.3139 - acc: 0.9056 - val_loss: 0.2199 - val_acc: 0.9281
Epoch 4/10
107/107 [==============================] - 80s 746ms/step - loss: 0.2258 - acc: 0.9330 - val_loss: 0.2631 - val_acc: 0.9062
Epoch 5/10
107/107 [==============================] - 80s 748ms/step - loss: 0.2054 - acc: 0.9336 - val_loss: 0.2168 - val_acc: 0.9219
Epoch 6/10
107/107 [==============================] - 80s 748ms/step - loss: 0.1848 - acc: 0.9424 - val_loss: 0.3004 - val_acc: 0.9094
Epoch 7/10
107/107 [==============================] - 81s 749ms/step - loss: 0.1687 - acc: 0.9515 - val_loss: 0.2496 - val_acc: 0.9312
```

## Image Classification using GNetTrainer

Create conda environment

```python
conda create -n GNetTrainer python=3.6
```

Activate conda environment

```python
conda activate GNetTrainer
```

Now all set to install the GNetTrainer Package 
```python
pip install GNetTrainer
```
To start the Magic. In the terminal type
```terminal
gnet
```


