#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 6.2
#  in conjunction with Tcl version 8.6
#    Nov 13, 2021 02:45:23 PM CST  platform: Windows NT
#    Nov 13, 2021 03:44:24 PM CST  platform: Windows NT
#    Nov 13, 2021 03:46:10 PM CST  platform: Windows NT
#    Nov 13, 2021 06:01:09 PM CST  platform: Windows NT

import sys
import os
import func
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top

def onBtnModifyClick_1():
    # print('GUI_support.onBtnModifyClick_1')
    # sys.stdout.flush()
    global img
    global text_get
    global fn
    img_list = os.listdir('./img')
    text_get = w.TEntry1.get()
    fn = img_list[int(text_get)]
    img_P = os.path.join('./img', fn)
    img_org = cv2.imread(img_P)    # bgr
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    

    # pos
    func._Pos(img, text_get)
    func._PlotPos(img, text_get)

def onBtnModifyClick_2():
    global src
    global res
    global dst
    if w.TEntry2.get() =='':
        dst = np.array([[65, 90], [95, 90], [80, 120]], np.float)
        H, W = 160, 190
    else:
        dst, H, W = func._ProcInput(w.TEntry2.get())
    src, res = func._Trans(img, text_get, dst, H, W)
    fig = plt.figure()
    plt.imshow(res)
    plt.show()    

def onBtnModifyClick_3():
    res_ = res.astype(np.float32)
    cv2.imwrite(os.path.join('./result', 'result_' + fn), cv2.cvtColor(res_, cv2.COLOR_RGB2BGR))
    print('Saved')

def onBtnModifyClick_4():
    inv = func._Inv(dst, src, img, res)
    inv_ = inv.astype(np.float32)
    cv2.imwrite(os.path.join('./result', 'inv_result_' + fn), cv2.cvtColor(inv_, cv2.COLOR_RGB2BGR))
    fig = plt.figure()
    plt.imshow(inv)
    plt.show() 

def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None

if __name__ == '__main__':
    import GUI
    GUI.vp_start_gui()





