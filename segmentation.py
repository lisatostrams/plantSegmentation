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
from scipy import ndimage

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.patches as mpatches
#%%

dirc = 'plaatjes'


ls = listdir(dirc)
ext = 'png jpg jpeg gif'
ls = [l for l in ls if l[-3:] in ext and 'mask' not in l and 'cleaning' not in l]
from statistics import mode
#
def mask_greens(img,greenthresh=90,redthresh=230,bluethresh=230):
    green = img[:,:,1]>greenthresh
    not_red = img[:,:,0] < redthresh
    not_blue = img[:,:,2] < bluethresh
    diffrg = abs(img[:,:,0] - img[:,:,1]) > 30
    diffgb = abs(img[:,:,1] - img[:,:,2]) < 230
    diffrb = abs(img[:,:,0] - img[:,:,2]) < 230
    color_less  = np.logical_and(diffrb,np.logical_and(diffrg,diffgb))
    return np.logical_and(np.logical_and(np.logical_and(green,not_red),not_blue),color_less)
    
def find_box(image):   
    label_image = label(image)
    area = []
    for region in regionprops(label_image):
        if region.area >= 400:
            minr, minc, maxr, maxc = region.bbox
            area.append((minr,minc,maxr,maxc,region.area))
    areasum = np.sum([a[4] for a in area])

    area = sorted(area, key=lambda item: abs(item[4]), reverse=True)[:10]
    area = np.array(area,dtype=float)
    for a in area:
        a[4]=np.float(a[4])/areasum
    area = [a for a in area if a[4]>.3]
    area = np.array(area,dtype=float)
    minr = np.min(area[:,0])
    minc = np.min(area[:,1])
    maxr = np.max(area[:,2])
    maxc = np.max(area[:,3])
    

    return (minr,minc,maxr,maxc),area

def clean_box(image,box):
    minr,minc,maxr,maxc = box
    xs,ys = np.where(image.astype('uint8')>0)
    oobx = np.logical_or(xs<minr,xs>maxr)
    ooby = np.logical_or(ys<minc,ys>maxc)
    oob = np.logical_or(oobx,ooby)
    xso = xs[oob]
    yso = ys[oob]
    image[xso,yso]=False
    return image
    

imgs = []
for f in ls:
    
    img1 = imageio.core.image_as_uint(imageio.imread(dirc+'/'+f))

    r,c,_ = img1.shape

    imgs.append(img1)
    #img1 = ndimage.gaussian_filter(img1,0.4)
    
    fig, ax = plt.subplots(2, 2,figsize=(18,12))
    ax[0,0].imshow(img1,cmap='gray')
    #ax[0,1].hist(img1[:,:,1].ravel(), bins=255, range=[0, 256])
    #ax[0,1].set_title('Green channel distribution')
    #ax[0,1].set_xlim(0, 256);
    twigs = np.sum(img1[:,:]>235,axis=2)

    ax[0,1].imshow(closing(twigs))

    img1 = img1[200:1500,:]

 
    greens = closing(mask_greens(img1,))
    greens=clear_border(greens)
    greens = closing(greens,square(10))
    (minr,minc,maxr,maxc),area = find_box(greens)
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='red', linewidth=2)
    ax[1,0].add_patch(rect)
    ax[1,0].plot([minc,minr],'b+',linewidth=20)
    ax[1,0].imshow(greens)
    greens=clean_box(greens,(minr,minc,maxr,maxc))
    greens = closing(greens,square(10))
    
    ax[1,1].imshow(greens)
    r,c = maxc - minc, maxr - minr
    pct = sum(greens.ravel())/(r*c)
    ax[1,1].set_title('{:.4f} percent of the box is plant'.format(pct*100))
    plt.show()
    #plt.savefig('plaatjes/cleaning_'+f)
    xs,ys = np.where(greens==False)
    img1[xs,ys]=255
    plt.imshow(img1)
    plt.title('{:.2f}% percent of the box is plant'.format(pct*100))
    plt.tight_layout()
    plt.savefig('plaatjes/masked_'+f,dpi=300)
#%%
#from skimage.feature import corner_harris, corner_subpix, corner_peaks
#import matplotlib.patches as mpatches
#
#
#
#from skimage.transform import (hough_line, hough_line_peaks,
#                               probabilistic_hough_line)
#from skimage.feature import canny
#h, theta, d = hough_line(image)
## Generating figure 1
#fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#ax = axes.ravel()
#
#ax[0].imshow(image, cmap=plt.cm.gray)
#ax[0].set_title('Input image')
#ax[0].set_axis_off()
#from skimage.draw import line
#
#
#
#ax[1].imshow(image, cmap=plt.cm.gray)
#for some, angle, dist in zip(*hough_line_peaks(h, theta, d)):
#    print(angle)
#    if((angle>-0.1 and angle < .1) or (angle>1.5 and angle<1.8)):
#        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
#        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
#        ax[1].plot((0, image.shape[1]), (y0, y1), '-r')
#ax[1].set_xlim((0, image.shape[1]))
#ax[1].set_ylim((image.shape[0], 0))
#ax[1].set_axis_off()
#ax[1].set_title('Detected lines')
#
#plt.tight_layout()
#plt.show()
#
#thresh = threshold_otsu(image)
#bw = closing(image > thresh, square(1))
##bw= closing(bw,square(3))
#plt.imshow(bw)
##contours = measure.find_contours(r, 0.1)
##
### Display the image and plot all contours found
##fig, ax = plt.subplots()
##ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)
##
##for n, contour in enumerate(contours):
##    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
##
##
## #remove artifacts connected to image border
#cleared = clear_border(bw)
#imshow(cleared)
#
#
## label image regions
#label_image = label(cleared)
#image_label_overlay = label2rgb(label_image, image=image)
#
#fig, ax = plt.subplots(figsize=(10, 6))
#ax.imshow(image_label_overlay)
#
#for region in regionprops(label_image):
#    # take regions with large enough areas
#    if region.area >= 4000:
#        print(region.area)
#        # draw rectangle around segmented coins
#        minr, minc, maxr, maxc = region.bbox
#        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
#                                  fill=False, edgecolor='red', linewidth=2)
#        ax.add_patch(rect)
#
#ax.set_axis_off()
#plt.tight_layout()
#plt.show()
