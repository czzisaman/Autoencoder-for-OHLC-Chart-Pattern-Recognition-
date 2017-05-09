#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 13:01:18 2017

@author: Marshall
"""
import os 

cwd = os.getcwd()
os.chdir('/Users/Marshall/Documents/PYTHON/Convol/cifar-10-batches-py/Images_15min_50_all_2')

#%% Convol Autoencoder
    
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(36, 52, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#%% Prepare Data 

import matplotlib.pyplot as plt
import scipy
import numpy as np

result = np.ndarray((0,36,52),dtype=np.uint8)  

i= 100  #this goes according to the file name as i saved in prepare_data_image
while True: 
    try:
        a = scipy.misc.imread("smallqq_" + str(i) + ".png")
        output = np.delete(a,np.s_[1:4], axis = 2)      
        output = np.reshape(output,(1,36,52))
        result = np.concatenate((result,output),axis =0)    
        i = i+50
    except FileNotFoundError:
        print('end of operation')        
        break

i= 115  #this goes according to the file name as i saved in prepare_data_image
while True: 
    try:
        a = scipy.misc.imread("smallqq_" + str(i) + ".png")
        output = np.delete(a,np.s_[1:4], axis = 2)      
        output = np.reshape(output,(1,36,52))
        result = np.concatenate((result,output),axis =0)    
        i = i+50
    except FileNotFoundError:
        print('end of operation')        
        break

    # "result" would be the final data  

#%% Manipulate Data    
import numpy as np

x_train = result[0:3300,:]
x_test =  result[3300:,:]   

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 36, 52, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 36, 52, 1))  # adapt this if using `channels_first` image data format

# tensorboard --logdir=/tmp/autoencoder

#%% Train
from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


#%% Plot result
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(36, 52))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i +1+ n)
    plt.imshow(decoded_imgs[i].reshape(36, 52))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#%% Plot with different d

n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#%% Save model
autoencoder.save('convol_autoencoder_v2.h5')


# the model we have is an autoencoder that reads the 50 bar 15 min charts, it's able
# to encode and decode any of them with reasonable sucess

#%% Load model
from keras.models import load_model

del autoencoder # deletes the existing model

# returns a compiled model
# identical to the previous one
autoencoder = load_model('convol_autoencoder_v2.h5')



