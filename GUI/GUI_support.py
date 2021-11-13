#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 6.2
#  in conjunction with Tcl version 8.6
#    Nov 13, 2021 02:45:23 PM CST  platform: Windows NT
#    Nov 13, 2021 03:44:24 PM CST  platform: Windows NT
#    Nov 13, 2021 03:46:10 PM CST  platform: Windows NT

import sys
import os
import func
import cv2

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

def onBtnModifyClick_2():
    print('GUI_support.onBtnModifyClick_2')
    sys.stdout.flush()

def onBtnModifyClick_1():
    # print('GUI_support.onBtnModifyClick_1')
    # sys.stdout.flush()
    img_list = os.listdir('./img')
    text_get = w.TEntry1.get()
    img_P = os.path.join('./img', img_list[int(text_get)])
    img_org = cv2.imread(img_P)    # bgr
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    # pos
    func._Pos(img, text_get)
    func._PlotPos(img, text_get)


def onBtnModifyClick_3():
    print('GUI_support.onBtnModifyClick_3')
    sys.stdout.flush()

def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None

if __name__ == '__main__':
    import GUI
    GUI.vp_start_gui()





