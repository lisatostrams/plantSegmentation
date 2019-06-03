#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:45:32 2019

@author: lisatostrams
"""

from os import listdir
dirc = 'raw'
ext = 'png jpg jpeg gif'
ls = listdir(dirc)
ls = [l for l in ls if l[-3:] in ext]

#%%

import matplotlib.pyplot as plt
import imageio
fig, ax = plt.subplots(2, 4,figsize=(20,12))
ax = ax.ravel()
i=0
for f in ls:
    img1 = imageio.core.image_as_uint(imageio.imread(dirc+'/'+f))
    ax[i].imshow(img1)
    i=i+1
plt.plot()

#%%

fig, ax = plt.subplots(2, 4,figsize=(20,12))
ax = ax.ravel()
i=0
for f in ls:
    img1 = imageio.core.image_as_uint(imageio.imread(dirc+'/'+f))
    ax[i].hist(img1[:,:,1].ravel(),bins=255)
    i=i+1
plt.plot()
#%%

fig, ax = plt.subplots(3, 2,figsize=(20,12))
ax = ax.ravel()
i=0
coords = ((200,400,150,600),(700,1200,900,1200))
for f in ls[:1]:
    for coord in coords:
        
        img1 = imageio.core.image_as_uint(imageio.imread(dirc+'/'+f))
        img1 = img1[coord]
        ax[i].imshow(img1)
        ax[i].set_title('Raw input {}'.format((i//3)+1))
        i=i+1
        red = img1[:,:,0].ravel()
        red = red[red<254]
        bins = max(red)-min(red)
        ax[i].hist(red,color='r',bins=bins,alpha=0.5)
        ax[i].set_title('Red, Green, Blue channels histograms input {}'.format((i//3)+1))
        blue = img1[:,:,2].ravel()
        blue = blue[blue<254]
        bins = max(blue)-min(blue)
        ax[i].hist(blue,color='b',bins=bins,alpha=0.5)
        green = img1[:,:,1].ravel()
        green = green[green<254]
        bins = max(green)-min(green)
        ax[i].hist(green,color='g',bins=bins,alpha=0.5)
        ax[i].set_ylim((0,30000))
        i=i+1
        diffrg = abs(img1[:,:,0] - img1[:,:,1]).ravel()
        diffgb = abs(img1[:,:,1] - img1[:,:,2]).ravel()
        diffbr = abs(img1[:,:,0] - img1[:,:,2]).ravel()
        ax[i].hist(diffrg,color='r',bins=bins,alpha=0.5)
        ax[i].hist(diffgb,color='g',bins=bins,alpha=0.5)
        ax[i].hist(diffbr,color='b',bins=bins,alpha=0.5)
        ax[i].set_ylim((0,100000))
        ax[i].set_title('Differences between RGB values')
        i=i+1
plt.savefig('histograms1and2.png',dpi=300)    
plt.plot()