#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:22:29 2017

@author: Marshall
"""

#%% Data Preparation 
# Need to first draw OHLC daily data into figures, and then read them into float64 image, 
# which can be used as input for autoencoder. 

import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc
from __future__ import print_function
import numpy as np
from utils import *
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt

# I downloaded lots of 15min data as csv and saved in a file, use terminal code to cat them

my_data = genfromtxt('merged.csv', delimiter=',')    # Select the source csv file here
data = np.delete(my_data, [0,1,6], axis = 1)  #This will be the OHLC data after clearing
size = data.shape


           
#%%  Generate and save images 

x = 115  # this is how many candles per chart
i = 50  # this is how many candles per chart

while x <= size[0]:
    opens = data[x-i:x,0]
    highs= data[x-i:x,1]
    lows = data[x-i:x,2]
    closes = data[x-i:x,3]
    
    fig, ax = plt.subplots()
    plt.axis('off')
    matplotlib.finance.candlestick2_ohlc(ax, opens, highs, lows, closes, width=1, colorup='k', colordown='r', alpha=0.75)
    fig.savefig("qq_" + str(x) + ".png")   # images are saved accordingly to the directory
    plt.close()
    x += i

#%% Shrinking the image

import PIL
from PIL import Image

i = 115
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

