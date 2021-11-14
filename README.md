# Affine Transform
NCKU Image Processing and Robot Vision course homework

## 專案目錄結構
```
Project
│   GUI.py
│   GUI_support.py
│   func.py
│   requirements.txt  
│   README.md      
│   ...    
└───img   
│   │   2018043072138985.jpg
│   │   ...
└───result   
│   │   result_2018043072138985.jpg
│   │   inv_result_2018043072138985.jpg
|   |   ...
└───npy
|   |   src_pos_0.npy
|   |   ...
└───ipynb 
```

## 前置工作
### 作業說明
* 目標\
透過影像處理的方式將圖中人臉以左右眼及鼻子之座標為準，\
轉移至設定好的模板上 (大小為160 x 190 pixels)

### 環境
* python 3.8
* Win 10

### 使用方式
1. 進入專案資料夾\
`cd [path/to/this/project]` 

2. 使用`pip install -r requirements.txt`安裝所需套件

3. 將欲處理的影像放入`./img`中

4. 執行GUI進行影像處理\
`python GUI.py`   
GUI介面說明如下：\
![Imgur](https://i.imgur.com/8QIDNBC.png)
(1) 輸入影像於`./img`中的順序(從0開始編號)
![Imgur](https://i.imgur.com/w4EpmMU.png)
(2) 加載影像並依序選取人像之右眼，左眼及鼻子座標；\
選取完畢後會顯示選取點之三角形範圍(編號為選取順序示意)，如下所示：\
![Imgur](https://i.imgur.com/KeOPg7c.png)
此外，該處會將選取的座標點以`src_pos_<圖片編號>.npy`的形式儲存至`./npy`
![Imgur](https://i.imgur.com/oxRvKdf.png)
(3) 輸入轉換目標點(與(2)選取順序相同)及模板大小；\
若無輸入則選用預設值`(65, 90)_(95, 90)_(80, 120)_(160, 190)`\
(4) 進行轉換，結束後會輸出成品如下：\
![Imgur](https://i.imgur.com/yoI22ZP.png)
(5) 將(4)之結果保存至`./result`中，命名規則為`result_<原圖檔名>.<原副檔名>`\
(6) 若有需要，可將(4)的結果再轉換為原坐標系，如下所示：
![Imgur](https://i.imgur.com/vrH9hb2.png)

### 操作影片
[![Imgur](https://i.imgur.com/GZHTlMo.png)](https://youtu.be/VB0ovF0sxiA)


## 程式碼說明
此處略過GUI設計說明。
### Image Input
```py
# GUI_support.py
img_list = os.listdir('./img')
# 取得./img中影像列表
text_get = w.TEntry1.get()
# 取得GUI輸入(此處為影像編號)
fn = img_list[int(text_get)]
# 選取影像之檔名
img_P = os.path.join('./img', fn)
img_org = cv2.imread(img_P)    # bgr
# 加載影像
img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
# 從BGR轉至RGB
# pos
func._Pos(img, text_get)
# 生成影像用以供使用者標記目標點
func._PlotPos(img, text_get)
# 畫上選取範圍
```
```py
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
```
```py
def _PlotPos(img, idx):
    img_c = np.copy(img)
    src = np.load('./npy/src_pos_' + idx + '.npy').astype(float)
    cv2.polylines(img_c, [src.astype(int)], True, (255, 0, 0), 2)
    # 取選取之左右眼及鼻子座標，畫出範圍於原影像上
    plt.imshow(img_c)
    plt.show()
```

### Get Affine Matrix of Affine Transform
```py
# func.py
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
```
各矩陣排列方式如下：
![Imgur](https://i.imgur.com/r44CcWJ.png)

### Apply Affine Transform with Affine Matrix
```py
# func.py
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
```

### Image output
```py
# GUI_support
cv2.imwrite(os.path.join('./result', 'result_' + fn), cv2.cvtColor(res_, cv2.COLOR_RGB2BGR))
# 將結果轉回BGR使用cv2儲存
```

## 轉換結果展示
2018043072138985.jpg
![Imgur](https://i.imgur.com/X7ZoIl0.jpg)
![Imgur](https://i.imgur.com/BrWQln1.jpg)
tom-cruise-vanessa-kirby-mission-impossible-fallout-1564649325.bmp
![Imgur](https://i.imgur.com/OvLf4Xl.png)
![Imgur](https://i.imgur.com/rbuPHSn.png)
TomHanksApr09.jpg
![Imgur](https://i.imgur.com/7H2w2sj.jpg)
![Imgur](https://i.imgur.com/eSiUeAA.jpg)
