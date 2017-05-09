#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 13:01:18 2017

@author: Marshall
"""

#%% Simple Autoencoder: with a single fully-connected neural layer as encoder and as decoder:
    # https://blog.keras.io/building-autoencoders-in-keras.html
    
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

# this is the size of our encoded representations
encoding_dim = 64  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(1872,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(1872, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')




#%% Prepare Data 
import matplotlib.pyplot as plt
import scipy
import numpy as np

'''
(Trial error Code when i was working)

a = scipy.misc.imread('qq_100.png')
plt.imshow(a)                              
output = np.delete(a,np.s_[1:4], axis = 2)      
output = np.reshape(output,(1,288,432))


b = scipy.misc.imread('qq_150.png')
plt.imshow(b)                              
output2 = np.delete(b,np.s_[1:4], axis = 2)      
output2 = np.reshape(output,(1,288,432))

x = output2[1:2]

arrays = [np.random.randn(3, 4) for _ in range(10)]
        
z = output[0,:]


zz = np.concatenate((output,output2),axis =0)
zzz = np.concatenate((output,zz),axis =0)
# need to concatenate all images' 3d data into one 
'''

result = np.ndarray((0,36,52),dtype=np.uint8)  #288*432 is image data struct

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
    # "result" would be the final data  

#%% Train    
from keras.datasets import mnist
import numpy as np

x_train = result[0:1500,:]
x_test =  result[1500:,:]   

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train,
                epochs=2000,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

#%% Plot result

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(36, 52))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(36, 52))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#%% Try shrinking the image to use less data
import PIL
from PIL import Image

i = 100 
while True:
    try:
        img = Image.open('qq_'+str(i)+'.png')
        img = img.resize((52,36), PIL.Image.ANTIALIAS)    # we want 54*36 data here
        img.save( "smallqq_" + str(i) + ".png")
        i += 50
    except FileNotFoundError:
        print('end of operation')        
        break
#        



