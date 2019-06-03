#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 23:25:26 2019

@author: lisatostrams
"""
import imageio
for i in range(0,8):
    img = imageio.imread('data/cnnmasks/mask_{}.png'.format(i),as_gray=True)
    img = (img-min(img.ravel()))/max(img.ravel())
    imageio.imsave('data/cnnmasks/mask_{}.png'.format(i),img)
