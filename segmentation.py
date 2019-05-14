#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:56:25 2019

@author: lisatostrams
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio

import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import cannyEdgeDetector as ced
from PIL import Image
from skimage import measure
from os import listdir
from skimage.morphology import closing
dirc = 'plaatjes'
ls = listdir(dirc)
ext = 'png jpg jpeg gif'
ls = [l for l in ls if l[-3:] in ext]
from statistics import mode
#
def mask_greens(img,greenthresh=100,redthresh=200,bluethresh=200):
    green = img[:,:,1]>greenthresh
    not_red = img[:,:,0] < redthresh
    not_blue = img[:,:,2] < bluethresh
    diffrg = abs(img[:,:,0] - img[:,:,1]) > 40
    diffgb = abs(img[:,:,1] - img[:,:,2]) < 220
    diffrb = abs(img[:,:,0] - img[:,:,2]) < 220
    color_less  = np.logical_and(diffrb,np.logical_and(diffrg,diffgb))
    return np.logical_and(np.logical_and(np.logical_and(green,not_red),not_blue),color_less)
    


imgs = []
for f in ls:
    img1 = imageio.core.image_as_uint(imageio.imread(dirc+'/'+f))
    r,c,_ = img1.shape

    imgs.append(img1)
    
    fig, ax = plt.subplots(1, 2,figsize=(12,6))
    ax[0].imshow(img1,cmap='gray')
    ax[1].hist(img1[:,:,1].ravel(), bins=255, range=[0, 256])
    ax[1].set_title('Green channel distribution')
    ax[1].set_xlim(0, 256);
    plt.show()
    greens = closing(mask_greens(img1,))
    plt.imshow(greens)

    pct = sum(greens.ravel())/(r*c)
    plt.title('{:.4f} percent of the image is plant'.format(pct*100))
    plt.savefig('plaatjes/masked_'+f)
##
#    
#def circle_points(resolution, center, radius):
#    """
#    Generate points which define a circle on an image.Centre refers to the centre of the circle
#    """   
#    radians = np.linspace(0, 2*np.pi, resolution)
#    c = center[1] + radius*np.cos(radians)#polar co-ordinates
#    r = center[0] + radius*np.sin(radians)
#    
#    return np.array([c, r]).T
## Exclude last point because a closed path should not have duplicate points
#cat = circle_points(200, [720, 250], 280)[:-1]
#human = circle_points(200,[700,750],280)[:-1]    
#renske = imgs[5]
#fig,ax = plt.subplots(1,1,figsize=(8,5))
#plt.imshow(renske)
#image_gray = color.rgb2gray(renske) 
#plt.imshow(renske)
#plt.plot(cat[:, 0], cat[:, 1], '--r', lw=2)
#plt.plot(human[:, 0], human[:, 1], '--g', lw=2)
#snake = seg.active_contour(image_gray, cat,alpha=0.002,beta=0.1)
#snake_hum = seg.active_contour(image_gray,human,alpha=0.002,beta=0.1)
#plt.plot(snake[:, 0], snake[:, 1], '-b', lw=1);
#plt.plot(snake_hum[:, 0], snake_hum[:, 1], '-b', lw=1);

