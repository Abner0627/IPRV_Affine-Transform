# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:45:57 2021

@author: Lab
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import colors

#%%
img_P = "./img/TomHanksApr09.jpg"
# img_P = "./img/tom-cruise-vanessa-kirby-mission-impossible-fallout-1564649325.bmp"
img = cv2.imread(img_P)    # bgr
h, w, _ = img.shape

fig = plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#%%
img_c = np.copy(img)
pts = np.array([[276, 173], [317, 179], [285, 226]], np.float32)  # eye_L, eye_R, nose
cv2.polylines(img_c, [pts.astype(int)], True, (255, 255, 0), 1)

fig = plt.figure()
plt.imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB))

#%%
# =============================================================================
# img_T = np.zeros([160, 190, 3], np.uint8)
# img_T.fill(200)
# 
# fig = plt.figure()
# plt.imshow(cv2.cvtColor(img_T, cv2.COLOR_BGR2RGB))
# =============================================================================

#%%
pts_tar = np.array([[65, 90], [95, 90], [80, 120]],  np.float32)

M = cv2.getAffineTransform(pts, pts_tar)
res = cv2.warpAffine(img, M, (160, 190))

fig = plt.figure()
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))