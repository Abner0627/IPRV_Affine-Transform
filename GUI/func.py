# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
import os 

# %% [markdown]
# ## Func

# %%
def _AffineMatrix(src, dst):
    src_ = np.vstack([src.T, np.array([1,1,1])])
    dst_ = np.vstack([dst.T, np.array([1,1,1])])
    M_ = np.dot(np.dot(dst_, src_.T), np.linalg.inv(np.dot(src_, src_.T)))
    return M_

def _Warp(img, M_, WH):
    W, H = WH[0], WH[1]
    src_y, src_x = np.indices(img.shape[:2])
    src_pts = np.stack((src_x.ravel(), src_y.ravel(), np.ones(src_y.size))).astype(int)
    dst_pts = np.round(np.dot(M_, src_pts)).astype(int)

    dst_img = np.zeros([W, H, 3])
    for c in range(3):
        for i in range(dst_pts.shape[-1]):
            dst_x, dst_y = dst_pts[0, i], dst_pts[1, i]
            src_u, src_v = src_pts[0, i], src_pts[1, i]
            if 0<=dst_x<H and 0<=dst_y<W:
                dst_img[dst_y, dst_x, c] = img[src_v, src_u, c]
    dst_img = dst_img.astype(int)  
    return dst_img  


def _PlotPos(img, idx):
    img_c = np.copy(img)
    src = np.load('src_pos_' + idx + '.npy').astype(float)
    cv2.polylines(img_c, [src.astype(int)], True, (255, 0, 0), 2)
    # fig = plt.figure()
    plt.imshow(img_c)
    plt.show()

def _Pos(img, idx):
    def on_press(event):
        L.append(np.array([int(event.xdata), int(event.ydata)]))
        if len(L)>=3: 
            plt.close()
        np.save('src_pos_' + idx + '.npy', np.array(L))

    fig = plt.figure()
    plt.imshow(img, animated= True)
    L = []
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show() 

def _ProcInput(in_str):
    warnings.warn("Wrong input format")
    B = in_str.split('_')
    C = [i.replace("(","") for i in B]
    D = [i.replace(")","") for i in C]
    E = [i.split(', ') for i in D]
    F = np.array(E).astype(float)
    return F[:2, :], F[3, 0], F[3, 1]

def _Trans(img, idx, dst, H, W):
    src = np.load('src_pos_' + idx + '.npy').astype(float)
    M = _AffineMatrix(src, dst)
    res = _Warp(img, M, (H, W))
    return src, res

def _Inv(dst, src, img, res):
    M_inv = _AffineMatrix(dst, src)
    W_org, H_org, _ = img.shape
    inv = cv2.warpAffine(res.astype(np.uint8), M_inv[:-1,:], (H_org, W_org))
    return inv




# # %% [markdown]
# # ## src

# # %%
# img_c = np.copy(img)
# src = np.array([[276, 173], [317, 179], [285, 226]], np.float32)  # eye_L, eye_R, nose
# cv2.polylines(img_c, [src.astype(int)], True, (255, 255, 0), 1)

# fig = plt.figure()
# plt.imshow(img_c)

# # %% [markdown]
# # ## dst

# # %%
# dst = np.array([[65, 90], [95, 90], [80, 120]],  np.float32)
# M = _AffineMatrix(src, dst)
# res = _Warp(img, M, (160, 190))

# fig = plt.figure()
# plt.imshow(res)         




# %%
