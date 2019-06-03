#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:56:25 2019

@author: lisatostrams
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio

from os import listdir
from skimage.morphology import closing

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import square
import matplotlib.patches as mpatches

dirc = 'plaatjes'


ls = listdir(dirc)
ext = 'png jpg jpeg gif'
ls = [l for l in ls if l[-3:] in ext and 'mask' not in l and 'cleaning' not in l]

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
    
for f in ls:
    
    img1 = imageio.core.image_as_uint(imageio.imread(dirc+'/'+f))

    r,c,_ = img1.shape

    fig, ax = plt.subplots(2, 2,figsize=(18,12))
    ax[0,0].imshow(img1,cmap='gray')
    twigs = np.sum(img1[:,:]>235,axis=2)

    ax[0,1].imshow(closing(twigs))

    img1 = img1[200:1500,:]
    imageio.imwrite('plaatjes/image_{}.png'.format(i),img1)
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
    imageio.imwrite('plaatjes/mask_{}.png'.format(i),greens*255)
    i=i+1
    
    
