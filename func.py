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
    """Get affine matrix of affine transform

    Args:
        src ([int]): coordinate point in source image (2x3)
        dst ([int]): coordinate point in target template (2x3)

    Returns:
        [float]: affine matrix
    """
    src_ = np.vstack([src.T, np.array([1,1,1])])
    dst_ = np.vstack([dst.T, np.array([1,1,1])])
    # 分別將目標座標(dst)與原座標(src)
    M_ = np.dot(np.dot(dst_, src_.T), np.linalg.inv(np.dot(src_, src_.T)))
    # 計算仿射矩陣M，假設目標座標矩陣為D；原座標矩陣為S
    # M = DS^T(SS^T)^-1
    return M_

def _Warp(img, M_, WH):
    """Apply affine transform with affine matrix, M_

    Args:
        img ([int]): source image
        M_ ([float]): affine matrix (3x3)
        WH ([tuple]): the width and height of template (W, H)

    Returns:
        [int]: the result image after affine transform
    """
    W, H = WH[0], WH[1]
    src_y, src_x = np.indices(img.shape[:2])
    # 取得所有像素點之x與y座標，並flatten之
    src_pts = np.stack((src_x.ravel(), src_y.ravel(), np.ones(src_y.size))).astype(int)
    # 排列成上述矩陣形式
    dst_pts = np.round(np.dot(M_, src_pts)).astype(int)
    # D = MS
    dst_img = np.zeros([W, H, 3])
    # 設定模板
    for c in range(3):
        for i in range(dst_pts.shape[-1]):
            dst_x, dst_y = dst_pts[0, i], dst_pts[1, i]
            src_u, src_v = src_pts[0, i], src_pts[1, i]
            # 取得在矩陣D與S中成對的座標(d_x, d_y), (s_u, s_v)
            if 0<=dst_x<H and 0<=dst_y<W:
                dst_img[dst_y, dst_x, c] = img[src_v, src_u, c]
                # 當轉換後座標(d_x, d_y)在模板範圍內，取對應原影像座標(s_u, s_v)之像素點
                # 若否，則定義該位置像素為0
    dst_img = dst_img.astype(int)  
    return dst_img  

def _PlotPos(img, idx):
    img_c = np.copy(img)
    src = np.load('./npy/src_pos_' + idx + '.npy').astype(float)
    cv2.polylines(img_c, [src.astype(int)], True, (255, 0, 0), 2)
    # 取選取之左右眼及鼻子座標，畫出範圍於原影像上
    plt.imshow(img_c)
    plt.show()

def _Pos(img, idx):
    def on_press(event):
        L.append(np.array([int(event.xdata), int(event.ydata)]))
        # 紀錄點選的座標點
        if len(L)>=3: 
            plt.close()
            # 當點選次數大於等於3時，關閉視窗
        np.save('./npy/src_pos_' + idx + '.npy', np.array(L))
        # 儲存紀錄座標點
    fig = plt.figure()
    plt.imshow(img, animated= True)
    L = []
    fig.canvas.mpl_connect('button_press_event', on_press)
    # 用動態圖的形式產生介面供使用者點選目標點
    plt.show() 

def _ProcInput(in_str):
    """Get coordinate point in target template from GUI type-in

    Args:
        in_str ([str]): GUI type-in

    Returns:
        dst ([int]): coordinate point in target template (2x3)
        H ([int]): the width of template
        W ([int]): the height of template
    """
    warnings.warn("Wrong input format")
    B = in_str.split('_')
    C = [i.replace("(","") for i in B]
    D = [i.replace(")","") for i in C]
    E = [i.split(', ') for i in D]
    # 分離輸入轉換目標點及模板大小分離
    F = np.array(E).astype(float)
    return F[:2, :], F[3, 0], F[3, 1]
    # dst, H, W

def _Trans(img, idx, dst, H, W):
    src = np.load('./npy/src_pos_' + idx + '.npy').astype(float)
    M = _AffineMatrix(src, dst)
    res = _Warp(img, M, (H, W))
    return src, res

def _Inv(dst, src, img, res):
    M_inv = _AffineMatrix(dst, src)
    W_org, H_org, _ = img.shape
    inv = cv2.warpAffine(res.astype(np.uint8), M_inv[:-1,:], (H_org, W_org))
    return inv