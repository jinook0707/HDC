# coding: UTF-8

"""
pyAnimalBehaviourCoder
An open-source software written in Python
  for tracking animal body and coding animal behaviour
  with collected video data. 

This program was coded and tested in macOS 10.13.

Jinook Oh, Cognitive Biology department, University of Vienna
October 2019.

Dependency:
    wxPython (4.0)
    NumPy (1.17)
    OpenCV (4.1)

------------------------------------------------------------------------
Copyright (C) 2019 Jinook Oh, W. Tecumseh Fitch
- Contact: jinook.oh@univie.ac.at, tecumseh.fitch@univie.ac.at

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------

------------------------------------------------------------------------
User input with mouse and keyboard
------------------------------------------------------------------------
* Left mouse click: Set the head direction by click-and-drag 
* Right mouse click: Set the head direction as -1 (when it's not available)
* Spacebar: Continue to proceed to next frames
* M: Toggle 'manual input' (Script will keep previously determined head direction until this is off)
* Cmd + S: Save CSV result data of head direction
* Cmd + Q: Quit the app

* If an user selects consecutive multiple cells (rows) in data sheet and set the head direction by click-and-drag on image,
all the selected rows will be updated as manual input.
Any column (such as head position, direction or any other column) can be selected for this operation.
------------------------------------------------------------------------

Changelog
------------------------------------------------------------------------
v.0.1: (2015-2016)
    - Initial development.
v.0.2: (2019.01)
    - Modification for 'Dove19' tracking.
    - Re-added refactored (from 2015 code) reivison module.
v.0.3: (2019.11-12)
    - Changed to pyABC from HDC.
    - Changed video handling with image file to direct handling 
    with video file, with videoRW.py.
    - Removed (commenting out for now) 'Dove19'.
    - Refactoring some variable names, function names, etc.
    - Added 'Macaque19' (Study by Hiroki Koda)
v.0.3.1: (2020.02)
    - Changed output data array from Numpy's character array 
    to structred array.
"""

import queue
from threading import Thread 
from os import getcwd, path
from sys import argv
from copy import copy
from time import time, sleep
from datetime import timedelta
from glob import glob
from random import shuffle

import wx, wx.adv, wx.grid
import wx.lib.scrolledpanel as SPanel 
import cv2
import numpy as np

from cv_proc import CVProc
#from reviseCSV import ReviseCSV
from videoRW import VideoRW
from fFuncNClasses import GNU_notice, get_time_stamp, writeFile, getWXFonts
from fFuncNClasses import load_img, add2gbs, setupStaticText, PopupDialog
from fFuncNClasses import updateFrameSize, receiveDataFromQueue, stopAllTimers
from fFuncNClasses import calcI2DIRatio 

DEBUG = False 
__version__ = "0.3.1"

#=======================================================================

class AnimalBehaviourCoderFrame(wx.Frame):
    """ Frame for AnimalBehaviourCoder 
        
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """
    def __init__(self):
        if DEBUG: print("AnimalBehaviourCoderFrame.__init__()")

        ### init frame
        wPos = [0, 25]
        wg = wx.Display(0).GetGeometry()
        wSz = (wg[2], int(wg[3]*0.85))
        wx.Frame.__init__(
              self,
              None,
              -1,
              "pyABCoder v.%s"%(__version__),
              pos = tuple(wPos),
              size = tuple(wSz),
              style=wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER|wx.MAXIMIZE_BOX),
              )
        self.SetBackgroundColour('#333333')

        ### set app icon
        self.tbIcon = wx.adv.TaskBarIcon(iconType=wx.adv.TBI_DOCK)
        icon = wx.Icon("icon.ico")
        self.tbIcon.SetIcon(icon)

        ##### [begin] setting up attributes -----
        self.th = None # thread
        self.q2m = queue.Queue() # queue from thread to main
        self.q2t = queue.Queue() # queue from main to thread 
        self.wPos = wPos # window position
        self.wSz = wSz # window size
        self.fonts = getWXFonts(initFontSz=8, numFonts=3)
        pi = self.setPanelInfo()
        self.pi = pi # pnael information
        self.gbs = {} # for GridBagSizer
        self.panel = {} # panels
        self.timer = {} # timers
        self.session_start_time = -1
        self.oData = [] # output data
        self.fPath = "" # folder path including frame images 
        self.flagVRec = False # whether to record analysis result video
        self.vRecSzR = 0.25 # ratio to the original frame size 
        self.isRunning = False # analysis is running by pressing spacebar
        self.isLBPressed = False # whether left mouse button is pressed or not
        self.flagBlockUI = False # block user input 
        ### description of parameters
        self.paramDesc = {} 
        d = "Number of iterations of morphologyEx (for reducing noise"
        d += " & minor features after absdiff from background image)."
        d += " [cv2.MORPH_OPEN operation]"
        self.paramDesc["bgsMExOIter"] = d
        d = "Number of iterations of morphologyEx (for reducing noise"
        d += " & minor features after absdiff from background image)."
        d += " [cv2.MORPH_CLOSE operation]"
        self.paramDesc["bgsMExCIter"] = d
        d = "Parameter for cv2.threshold before edge detection."
        self.paramDesc["bgsThres"] = d
        d = "Parameters for hysteresis threshold of Canny function"
        d += " (in edge detection of difference from background image)."
        self.paramDesc["cannyTh"] = d
        d = "Minimum contour size (width + height) in recognizing"
        d += " contours of detected edges."
        self.paramDesc["contourTh"] = d
        d = "Lower and upper threshold for recognizing a motion in"
        d += " a frame. Threshold value is a square root of"
        d += " sum(different_pixel_values)/255."
        self.paramDesc["motionTh"] = d
        d = "Length (in pixels) of head direction line to draw."
        self.paramDesc["hdLineLen"] = d
        d = "If head direction differece is over this threshold,"
        d += " reject the calculated head direction and copy the previous"
        d += " frame's head direction."
        self.paramDesc["uDegTh"] = d 
        d = "Number of clusters for k-means clustering."
        self.paramDesc["uNKMC"] = d 
        # animal experiment cases
        self.animalECaseChoices = [
                            'Macaque19',
                            'Marmoset04', 
                            'Rat05', 
                            #'Dove19'
                            ]
        # current animal experiment case
        self.animalECase = self.animalECaseChoices[0]
        # set corresponding parameters
        self.setAECaseParam() 
        # display image type choice
        self.dispImgTypeChoices = ["RGB", "Greyscale(Diff)"] #"Greyscale(Edge)"
        # current display image type
        self.dispImgType = self.dispImgTypeChoices[0]
        self.frameImgFNformat = "f%06i.jpg"
        self.lpWid = [] # wx widgets in left panel
        self.ratFImgDispImg = None # ratio between frame image and 
          # display image on app
        self.flagContManualInput = False # continuous manual input
        self.dataGridSelectedCells = []
        self.setDataCols() # set ouput data columns (self.dataCols),
          # initival values (self.dataInitVal) and column indices 
        self.cv_proc = CVProc(self) # computer vision processing module
        self.vRW = VideoRW(self) # for reading/writing video file
        ##### [end] setting up attributes -----

        ### create panels
        for pk in pi.keys():
            self.panel[pk] = SPanel.ScrolledPanel(
                                                  self,
                                                  name="%s_panel"%(pk),
                                                  pos=pi[pk]["pos"],
                                                  size=pi[pk]["sz"],
                                                  style=pi[pk]["style"],
                                                 )
            self.panel[pk].SetBackgroundColour(pi[pk]["bgCol"])

        ##### [begin] set up top panel interface -----
        tpSz = pi["tp"]["sz"]
        vlSz = (-1, 20) # size of vertical line separator
        self.gbs["tp"] = wx.GridBagSizer(0,0)
        row = 0; col = 0
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="Analyze video",
                        name="analyze_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        '''
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="Revise analyzed results (CSV file)",
                        name="revise_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        '''
        col += 1 
        add2gbs(self.gbs["tp"],
                wx.StaticLine(self.panel["tp"],
                              -1,
                              size=vlSz,
                              style=wx.LI_VERTICAL),
                (row,col),
                (1,1)) # vertical line separator
        col += 1
        sTxt = setupStaticText(
                            self.panel["tp"],
                            "Animal: ",
                            font=self.fonts[2],
                            )
        add2gbs(self.gbs["tp"], sTxt, (row,col), (1,1))
        col += 1
        cho = wx.Choice(
                            self.panel["tp"],
                            -1,
                            name="animalECase_cho",
                            choices=self.animalECaseChoices,
                       )
        cho.Bind(wx.EVT_CHOICE, self.onChoice)
        cho.SetSelection(self.animalECaseChoices.index(self.animalECase))
        add2gbs(self.gbs["tp"], cho, (row,col), (1,1))
        col += 1
        add2gbs(self.gbs["tp"],
                wx.StaticLine(self.panel["tp"],
                              -1,
                              size=vlSz,
                              style=wx.LI_VERTICAL),
                (row,col),
                (1,1)) # vertical line separator
        col += 1
        sTxt = setupStaticText(
                            self.panel["tp"],
                            "Display-image: ",
                            font=self.fonts[2],
                            )
        add2gbs(self.gbs["tp"], sTxt, (row,col), (1,1))
        col += 1
        cho = wx.Choice(
                            self.panel["tp"],
                            -1,
                            name="imgType_cho",
                            choices=self.dispImgTypeChoices, 
                       )
        cho.Bind(wx.EVT_CHOICE, self.onChoice)
        add2gbs(self.gbs["tp"], cho, (row,col), (1,1))
        col += 1
        add2gbs(self.gbs["tp"],
                wx.StaticLine(self.panel["tp"],
                              -1,
                              size=vlSz,
                              style=wx.LI_VERTICAL),
                (row,col),
                (1,1)) # vertical line separator
        col += 1
        chk = wx.CheckBox(
                            self.panel['tp'], 
                            id=-1, 
                            label='Continuous manual input',
                            name="contManualInput_chk",
                            style=wx.CHK_2STATE,
                         )
        chk.Bind(wx.EVT_CHECKBOX, self.onCheckBox)
        chk.SetForegroundColour('#CCCCCC')
        add2gbs(self.gbs["tp"], chk, (row,col), (1,1))
        nCol = int(col) # number columns
        row += 1; col = 0 # new row
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="Quit",
                        name="quit_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["tp"], btn, (row,col), (1,2))
        col += 2
        sTxt = setupStaticText(
                            self.panel["tp"],
                            "0:00:00",
                            font=self.fonts[2],
                            name="ssTime_sTxt",
                            ) # elapsed time since session starts
        sTxt.SetBackgroundColour('#000000')
        sTxt.SetForegroundColour('#CCCCFF')
        add2gbs(self.gbs["tp"], sTxt, (row,col), (1,2))
        col += 2 
        sTxt = setupStaticText(
                            self.panel["tp"],
                            "FPS",
                            font=self.fonts[2],
                            name="fps_sTxt",
                            )
        sTxt.SetForegroundColour('#cccccc')
        add2gbs(self.gbs["tp"], sTxt, (row,col), (1,2))
        col += 2 
        sTxt = setupStaticText(
                            self.panel["tp"],
                            "FOLDER NAME",
                            font=self.fonts[2],
                            name="fp_sTxt",
                            )
        sTxt.SetForegroundColour('#ffffff')
        add2gbs(self.gbs["tp"], sTxt, (row,col), (1,nCol-col))
        self.panel["tp"].SetSizer(self.gbs["tp"])
        self.gbs["tp"].Layout()
        self.panel["tp"].SetupScrolling()
        ##### [end] set up top panel interface -----
        

        ##### [begin] set up left panel interface -----
        cho = wx.FindWindowByName("animalECase_cho", self.panel["tp"])
        self.initLPWidgets() # left panel widgets
        ##### [end] set up left panel interface -----
        
        ##### [begin] set up image panel interface -----
        ### for displaying frame image
        self.dispImg_sBmp_sz = pi['ip']['sz'] 
        self.dispImg_sBmp = wx.StaticBitmap(
                                        self.panel['ip'], 
                                        -1, 
                                        wx.NullBitmap, 
                                        (0,0), 
                                        pi['ip']['sz']
                                        )
        self.dispImg_sBmp.Bind(wx.EVT_ERASE_BACKGROUND, lambda evt: None) 
        self.dispImg_sBmp.Bind(wx.EVT_LEFT_DOWN, self.onMLBD_dispImg)
        self.dispImg_sBmp.Bind(wx.EVT_LEFT_UP, self.onMLBU_dispImg)
        self.dispImg_sBmp.Bind(wx.EVT_MOTION, self.onMMove_dispImg)
        self.dispImg_sBmp.Bind(wx.EVT_RIGHT_UP, self.onMRBU_dispImg)
        ##### [end] set up image panel interface -----

        ##### [begin] set up right panel interface -----
        nCol = 2
        self.gbs["rp"] = wx.GridBagSizer(0,0)
        row = 0; col = 0
        self.dataGrid = Grid(self.panel["rp"], np.empty((0,0), dtype=np.uint8))
        self.Bind(wx.grid.EVT_GRID_CELL_CHANGED, self.onDataGridCellChanged)
        self.Bind(wx.grid.EVT_GRID_RANGE_SELECT, self.onDataGridCellsSelected)
        add2gbs(self.gbs["rp"], self.dataGrid, (row,col), (1,nCol))
        row += 1; col = 0
        btn = wx.Button(self.panel["rp"],
                        -1,
                        label="to the next frame",
                        name="nextFrame_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["rp"], btn, (row,col), (1,1))
        col += 1
        btn = wx.Button(self.panel["rp"],
                        -1,
                        label="Move to the selected frame",
                        name="jump2frame_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["rp"], btn, (row,col), (1,1))
        col += 1
        sTxt = setupStaticText(
                            self.panel["rp"],
                            "-",
                            font=self.fonts[2],
                            name="navProg_sTxt",
                            )
        sTxt.SetForegroundColour('#cccccc')
        add2gbs(self.gbs["rp"], sTxt, (row,col), (1,1))
        self.panel["rp"].SetSizer(self.gbs["rp"])
        self.gbs["rp"].Layout()
        self.panel["rp"].SetupScrolling()
        ##### [end] set up right panel interface -----

        ### set up menu
        menuBar = wx.MenuBar()
        pyABCMenu = wx.Menu()
        quit = pyABCMenu.Append(wx.Window.NewControlId(), item="Quit\tCTRL+Q")
        menuBar.Append(pyABCMenu, "&pyABC")
        self.SetMenuBar(menuBar) 

        ### keyboard binding
        exitId = wx.NewIdRef(count=1)
        saveId = wx.NewIdRef(count=1)
        spaceId = wx.NewIdRef(count=1) # for running analysis
        miId = wx.NewIdRef(count=1) # for continuous manual input
        nextFrameId = wx.NewIdRef(count=1) # for loading one frame only
        self.Bind(wx.EVT_MENU, self.onClose, id=exitId)
        self.Bind(wx.EVT_MENU, self.onSave, id=saveId) 
        self.Bind(wx.EVT_MENU, self.onSpace, id=spaceId)
        self.Bind(wx.EVT_MENU,
                  lambda event: self.onCheckBox(event, "contManualInput_chk"),
                  id=miId)
        self.Bind(wx.EVT_MENU, 
                  lambda event: self.onRight(True),
                  id=nextFrameId)
        accel_tbl = wx.AcceleratorTable([
                        (wx.ACCEL_CMD,  ord('Q'), exitId ), 
                        (wx.ACCEL_CMD,  ord('S'), saveId ),
                        (wx.ACCEL_NORMAL, wx.WXK_SPACE, spaceId),
                        (wx.ACCEL_NORMAL,  ord('M'), miId), 
                        (wx.ACCEL_CMD, wx.WXK_RIGHT, nextFrameId),
                        # [Deprecated] navigation with Left/Right key
                        #(wx.ACCEL_SHIFT,  wx.WXK_RIGHT, right_btnId ), 
                        #(wx.ACCEL_SHIFT,  wx.WXK_LEFT, left_btnId ), 
                        #(wx.ACCEL_ALT,  wx.WXK_RIGHT, rightJump_btnId ), 
                        #(wx.ACCEL_ALT,  wx.WXK_LEFT, leftJump_btnId ),
                        #(wx.ACCEL_CMD, wx.WXK_LEFT, leftJumpFurther_btnId), 
                        #(wx.ACCEL_CMD, wx.WXK_RIGHT, rightJumpFurther_btnId),
                                        ]) 
        self.SetAcceleratorTable(accel_tbl) 

        ### set up status-bar
        self.statusbar = self.CreateStatusBar(1)
        self.sbBgCol = self.statusbar.GetBackgroundColour()
        self.timer["sb"] = None
        
        updateFrameSize(self, wSz)
        self.Bind( wx.EVT_CLOSE, self.onClose )

    #-------------------------------------------------------------------
   
    def setPanelInfo(self):
        """ Set up panel information.
        
        Args:
            None
        
        Returns:
            pi (dict): Panel information.
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.setPanelInfo()")

        wSz = self.wSz # window size
        pi = {} # panel information to return
        # top panel for buttons, etc
        pi["tp"] = dict(pos=(0, 0), 
                        sz=(wSz[0], 80), 
                        bgCol="#666666", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        tpSz = pi["tp"]["sz"]
        # left side panel for setting parameters
        pi["lp"] = dict(pos=(0, tpSz[1]), 
                        sz=(250, wSz[1]-tpSz[1]), 
                        bgCol="#333333", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        lpSz = pi["lp"]["sz"]
        # right side panel for displaying coded data 
        pi["rp"] = dict(pos=(wSz[0]-400, tpSz[1]), 
                        sz=(400, lpSz[1]), 
                        bgCol="#333333", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        rpSz = pi["rp"]["sz"]
        # panel for displaying frame image 
        pi["ip"] = dict(pos=(lpSz[0], tpSz[1]), 
                        sz=(wSz[0]-lpSz[0]-rpSz[0], lpSz[1]), 
                        bgCol="#000000", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        return pi
   
    #-------------------------------------------------------------------

    def setDataCols(self):
        """ Set output data columns depending on animal experiment case 

        Args: None

        Returns: None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.setDataCols()")

        if self.animalECase in ["Marmoset04", "Macaque19", "Rat05"]:
            self.dataCols = [
                             "hD", # head direction 
                             "mHD", # head direction is manually fixed
                             "hPosX", # head position (x)
                             "hPosY", # head position (y) 
                             "mHPos", # head position is manually fixed 
                             "bPosX", # body postion (x) 
                             "bPosY", # body position (y)
                             "remarks", 
                            ] # data columns 
            self.dataInitVal = [
                                "None", # hD 
                                "False", # mHD 
                                "None", # hPosX 
                                "None", # hPosY 
                                "False", # mHPos 
                                "None", # bPosX 
                                "None", # bPosY 
                                "None", # remarks
                               ] # initial values for data columns
            ### store indices for each data column
            self.hdi = self.dataCols.index("hD")
            self.mhdi = self.dataCols.index("mHD")
            self.hxi = self.dataCols.index("hPosX")
            self.hyi = self.dataCols.index("hPosY")
            self.mhpi = self.dataCols.index("mHPos")
            self.bxi = self.dataCols.index("bPosX")
            self.byi = self.dataCols.index("bPosY")
            self.dataStruct = [
                     ('hD', (np.str_, 4)), # head direction
                     ('mHD', (np.str_, 5)), # head direction is manually fixed
                     ('hPosX', (np.str_, 4)), # head position (x) 
                     ('hPosY', (np.str_, 4)), # head position (y) 
                     ('mHPos', (np.str_, 5)), # head position is manually fixed
                     ('bPosX', (np.str_, 4)), # base position (x) 
                     ('bPosY', (np.str_, 4)), # base position (y) 
                     ('remarks', (np.str_, 20)), # remakrs
                     ] # data types for numpy structured array
       
        '''
        elif self.animalECase == "Dove19":
            self.dataCols = [
                             "hD", # head direction 
                             "mHD", # head direction is manually fixed
                             "bD", # body direction 
                             "hPosX", # head position (x)
                             "hPosY", # head position (y) 
                             "mHPos", # head position is manually fixed 
                             "bPosX", # body postion (x) 
                             "bPosY", # body position (y) 
                             "b1PosX", # body position-1 (x) 
                             "b1PosY", # body position-1 (y) 
                             "b2PosX", # body position-2 (x) 
                             "b2PosY", # body position-2 (y) 
                            ] # data columns
            self.dataInitVal = [
                                "None", # hD 
                                "False", # mHD 
                                "None", # bD 
                                "None", # hPosX 
                                "None", # hPosY 
                                "False", # mHPos 
                                "None", # bPosX 
                                "None", # bPosY 
                                "None", # b1PosX 
                                "None", # b1PosY 
                                "None", # b2PosX 
                                "None", # b2PosY 
                               ] # initial values for data columns
            ### store indices for each data column
            self.hdi = self.dataCols.index("hD")
            self.mhdi = self.dataCols.index("mHD")
            self.bdi = self.dataCols.index("bD")
            self.hxi = self.dataCols.index("hPosX")
            self.hyi = self.dataCols.index("hPosY")
            self.mhpi = self.dataCols.index("mHPos")
            self.bxi = self.dataCols.index("bPosX")
            self.byi = self.dataCols.index("bPosY")
            self.b1xi = self.dataCols.index("b1PosX")
            self.b1yi = self.dataCols.index("b1PosY")
            self.b2xi = self.dataCols.index("b2PosX")
            self.b2yi = self.dataCols.index("b2PosY")
            '''
        
    #-------------------------------------------------------------------

    def initLPWidgets(self):
        """ initialize wx widgets in left panel

        Args: None

        Returns: None
        """ 
        if DEBUG: print("AnimalBehaviourCoderFrame.initLPWidgets()")

        for i, w in enumerate(self.lpWid):
        # through widgets in left panel
            self.gbs['lp'].Detach(w) # detach grid from gridBagSizer
            w.Destroy() # destroy the widget

        ### make a list to produce widgets with parameter keys
        lst = list(self.aecParam.keys()) # list of keys for parameter items 
        uLst = [] # this parameter items will be more likely modified by users.
        for i in range(len(lst)):
            if lst[i][0] == 'u':
                uLst.append(str(lst[i]))
                lst[i] = None
        while None in lst: lst.remove(None)
        lst = sorted(lst) + ["---"] + sorted(uLst)

        ##### [begin] set up left panel interface -----
        self.gbs["lp"] = wx.GridBagSizer(0,0)
        self.lpWid = [] # wx widgets in left panel
        lpSz = self.pi["lp"]["sz"]
        hlSz = (int(lpSz[0]*0.9), -1) # size of horizontal line
        row = 0; col = 0
        for key in lst:
            if key == "---":
                sLine = wx.StaticLine(self.panel["lp"],
                                      -1,
                                      size=hlSz,
                                      style=wx.LI_HORIZONTAL)
                sLine.SetForegroundColour('#CCCCCC')
                self.lpWid.append(sLine)
                add2gbs(self.gbs["lp"], sLine, (row,col), (1,3))
                row += 1; col = 0
                continue
            p = self.aecParam[key]
            sTxt = setupStaticText(
                            self.panel["lp"],
                            label=key,
                            font=self.fonts[1],
                            )
            sTxt.SetForegroundColour('#CCCCCC')
            self.lpWid.append(sTxt)
            add2gbs(self.gbs["lp"], sTxt, (row,col), (1,1))
            col += 1
            btn = wx.Button(self.panel["lp"],
                            -1,
                            label="?",
                            name="%s_help_btn"%(key),
                            size=(25,-1))
            btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
            self.lpWid.append(btn)
            add2gbs(self.gbs["lp"], btn, (row,col), (1,1))
            col += 1
            val = str(p["value"]).strip("[]")
            txt = wx.TextCtrl(self.panel["lp"], 
                              -1, 
                              value=val, 
                              name="%s_txt"%(key))
            self.lpWid.append(txt)
            add2gbs(self.gbs["lp"], txt, (row,col), (1,1))
            row += 1; col = 0
        ###
        btn = wx.Button(self.panel["lp"],
                        -1,
                        label="Analyze changed parameters",
                        name="applyParam_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        self.lpWid.append(btn)
        add2gbs(self.gbs["lp"], btn, (row,col), (1,3))
        ###
        self.panel["lp"].SetSizer(self.gbs["lp"])
        self.gbs["lp"].Layout()
        self.panel["lp"].SetupScrolling()
        ##### [end] set up left panel interface -----

    #-------------------------------------------------------------------
    
    def setAECaseParam(self):
        """ Set up parameters for the current animal experiment case.
        * Parameter key starts with a letter 'u' means that 
          it will probably modified by users more frequently.

        Args: None

        Returns: None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.setAECaseParam()") 
        
        if self.animalECase == "":
            return

        elif self.animalECase == "Marmoset04":
        # Common marmoset monkey experiment in 2004
        # Reber, Slipogor et al
            self.aecParam = {}
            self.aecParam["bgsMExOIter"] = dict(value=8) 
            self.aecParam["bgsMExCIter"] = dict(value=8) 
            self.aecParam["bgsThres"] = dict(value=60)
            self.aecParam["cannyTh"] = dict(value=[150, 150]) 
            self.aecParam["contourTh"] = dict(value=50)
            self.aecParam["motionTh"] = dict(value=[35, 100])
            self.aecParam["hdLineLen"] = dict(value=50)
            self.aecParam["uDegTh"] = dict(value=20)
        
        elif self.animalECase == "Macaque19":
        # Macaque monkey experiment in 2019 
        # Koda et al
            subjNames = ["PoCo", "PiGe", "Lutku"]
            subj = subjNames[2]
            d = "Head rect size"
            self.paramDesc["uHRSz"] = d
            d = "HSV color min values to detect bluish wooden panel."
            self.paramDesc["uCol0Min"] = d
            d = "HSV color max values to detect bluish wooden panel."
            self.paramDesc["uCol0Max"] = d
            d = "HSV color min values to detect macaque's head."
            self.paramDesc["uCol1Min"] = d
            d = "HSV color max values to detect macaque's head."
            self.paramDesc["uCol1Max"] = d
            d = "HSV color min values to detect macaque's head."
            d += ", when screen color was chaning macaque's face color."
            self.paramDesc["uCol2Min"] = d
            d = "HSV color max values to detect macaque's head."
            d += ", when screen color was chaning macaque's face color."
            self.paramDesc["uCol2Max"] = d
            d = "HSV color min values to detect macaque's face."
            self.paramDesc["uCol3Min"] = d
            d = "HSV color max values to detect macaque's face."
            self.paramDesc["uCol3Max"] = d
            d = "HSV color min values to detect macaque's face"
            d += ", when screen color was chaning macaque's face color."
            self.paramDesc["uCol4Min"] = d
            d = "HSV color max values to detect macaque's face"
            d += ", when screen color was chaning macaque's face color."
            self.paramDesc["uCol4Max"] = d
            self.aecParam = {}
            if subj == "PoCo": self.aecParam["uHRSz"] = dict(value=0.5) 
            elif subj == "PiGe": self.aecParam["uHRSz"] = dict(value=0.45) 
            elif subj == "Lutku": self.aecParam["uHRSz"] = dict(value=0.5) 
            self.aecParam["bgsMExOIter"] = dict(value=1)
            self.aecParam["bgsMExCIter"] = dict(value=-1)
            self.aecParam["bgsThres"] = dict(value=30)
            self.aecParam["cannyTh"] = dict(value=[10, 30])
            self.aecParam["contourTh"] = dict(value=50)
            self.aecParam["motionTh"] = dict(value=[35, 300])
            self.aecParam["hdLineLen"] = dict(value=150)
            ### bluish color of wooden panel below macaque's head 
            self.aecParam["uCol0Min"] = dict(value=[60,0,0])
            self.aecParam["uCol0Max"] = dict(value=[100,150,150])
            ### brownish color of macaque's head
            self.aecParam["uCol1Min"] = dict(value=[0,100,30])
            self.aecParam["uCol1Max"] = dict(value=[20,255,120])
            ### brownish color of macaque's head when screen changes its color
            self.aecParam["uCol2Min"] = dict(value=[0,100,30])
            self.aecParam["uCol2Max"] = dict(value=[20,255,130])
            ### pinkish color of macaque's face
            if subj == "PoCo":
                self.aecParam["uCol3Min"] = dict(value=[0,100,150])
                self.aecParam["uCol3Max"] = dict(value=[10,255,230])
            elif subj == "PiGe":
                self.aecParam["uCol3Min"] = dict(value=[0,120,150])
                self.aecParam["uCol3Max"] = dict(value=[12,255,230])
            elif subj == "Lutku":
                self.aecParam["uCol3Min"] = dict(value=[0,140,100])
                self.aecParam["uCol3Max"] = dict(value=[10,255,220])
            ### purplish color of macaque's face when screen changes its color 
            #self.aecParam["uCol3Min"] = dict(value=[150,120,100])
            self.aecParam["uCol4Min"] = dict(value=[150,120,120])
            self.aecParam["uCol4Max"] = dict(value=[200,255,230])

        elif self.animalECase == "Rat05": 
        # Rat tracking in 2005 (for testing purpose) 
            self.aecParam = {}
            self.aecParam["bgsMExOIter"] = dict(value=2)
            self.aecParam["bgsThres"] = dict(value=50)
            self.aecParam["cannyTh"] = dict(value=[150, 150])
            self.aecParam["contourTh"] = dict(value=5)
            self.aecParam["motionTh"] = dict(value=[25, 100])
            self.aecParam["hdLineLen"] = dict(value=30)
            self.aecParam["uDegTh"] = dict(value=30)
            self.aecParam["uNKMC"] = dict(value=4)
        
        '''
        elif self.animalECase == "Dove19": 
        # Dove tracking in 2019
            ### params used for dove19 experiment
            d = "Size threshold to find distal part of head."
            self.paramDesc["uSzTh4DH"] = d
            d = "Size threshold to find proximate part of head."
            self.paramDesc["uSzTh4DH"] = d
            d = "Minimum distance between valid bPos and hPos."
            self.paramDesc["uBHDistMin"] = d
            d = "Minimum size to detect dove."
            self.paramDesc["uBlobSzMin"] = d
            d = "Maximum size to detect dove."
            self.paramDesc["uBlobSzMax"] = d
            d = "Threshold to determine one of two ways to determine"
            d += " the calculation method for head-direction"
            d += " calculation. This value ranges from 0.0 to 1.0."
            d += " Low value means the head is out of body line."
            d += " High value means the head is tucked in the body line."
            self.paramDesc["uHDCalcTh"] = d
            self.aecParam = {}
            self.aecParam["bgsMExOIter"] = dict(value=1)
            self.aecParam["bgsThres"] = dict(value=20)
            self.aecParam["cannyTh"] = dict(value=[150, 150])
            self.aecParam["contourTh"] = dict(value=30)
            self.aecParam["motionTh"] = dict(value=[20, 500])
            self.aecParam["hdLineLen"] = dict(value=50)
            self.aecParam["uDegTh"] = dict(value=45)
            self.aecParam["uSzTh4DH"] = dict(value=75)
            self.aecParam["uSzTh4PH"] = dict(value=400)
            self.aecParam["uBHDistMin"] = dict(value=3)
            self.aecParam["uBlobSzMin"] = dict(value=30000)
            self.aecParam["uBlobSzMax"] = dict(value=50000)
            self.aecParam["uHDCalcTh"] = dict(value=0.5)
        '''
        
        ### add description to the dictionary
        for k in self.aecParam.keys():
            self.aecParam[k]["desc"] = self.paramDesc[k]
    
    #-------------------------------------------------------------------

    def onButtonPressDown(self, event, objName=""):
        """ wx.Butotn was pressed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate the button press
              of the button with the given name. 
        
        Returns:
            None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.onButtonPressDown()")

        if self.flagBlockUI: return 

        if objName == '':
            obj = event.GetEventObject()
            objName = obj.GetName()
        else:
            obj = wx.FindWindowByName(objName, self.panel["tp"])

        if not obj.IsEnabled(): return

        if objName == "analyze_btn":
            self.startStopAnalyzeVideo()
        elif objName == "quit_btn":
            self.onClose(None)
        elif objName == "nextFrame_btn":
            self.onRight(True)
        elif objName == "jump2frame_btn":
            self.jumpToFrame()
        elif objName == "applyParam_btn":
            self.applyChangedParam()
        elif objName.endswith("_help_btn"):
            key = objName.replace("_help_btn", "")
            msg = self.aecParam[key]["desc"]
            wx.MessageBox(msg, "Info", wx.OK|wx.ICON_INFORMATION) 
        '''
        elif objName == "revise_btn":
            self.dataRevision()
        '''

    #-------------------------------------------------------------------

    def onChoice(self, event, objName=""):
        """ wx.Choice was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate wx.Choice event 
                with the given name. 
        
        Returns:
            None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.onChoice()")

        if self.flagBlockUI: return 

        if objName == "":
            obj = event.GetEventObject()
            objName = obj.GetName()
        else:
        # funcion was called by some other function without wx.Event
            obj = wx.FindWindowByName(objName, self.panel["tp"])
            
        objVal = obj.GetString(obj.GetSelection()) # text of chosen option
        
        if objName == "animalECase_cho":
        # animal experiment case was chosen
            self.animalECase = objVal
            self.setAECaseParam() # set animal experiment case parameters
            self.initLPWidgets() # initialize left panel
            self.proc_img() # display frame image

        if objName == "imgType_cho":
        # display image type changed
            self.dispImgType = objVal 
            self.proc_img()

    #-------------------------------------------------------------------

    def onCheckBox(self, event, objName=""):
        """ wx.CheckBox was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate wx.CheckBox event 
                with the given name. 
        
        Returns: None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.onCheckBox()")

        if self.flagBlockUI: return 
        
        if objName == "":
            obj = event.GetEventObject()
            objName = obj.GetName()
            isChkBoxEvent = True 
        else:
        # funcion was called by some other function without wx.Event
            obj = wx.FindWindowByName(objName, self.panel["tp"])
            isChkBoxEvent = False 
        
        objVal = obj.GetValue() # True/False 

        if objName == "contManualInput_chk":
            if isChkBoxEvent:
                self.flagContManualInput = objVal
            else: # called by hotkey
                val = not objVal # opposite of the current checkbox value
                ### update checkbox & flag
                obj.SetValue(val) 
                self.flagContManualInput = val
            self.cv_proc.last_motion_frame = self.vRW.currFrame.copy()
    
    #-------------------------------------------------------------------
       
    def onRight(self, isLoadingOneFrame=False):
        """ Navigate forward 
        
        Args:
            isLoadingOneFrame (bool): whether load only one frame, 
              False by defualt.

        Returns:
            None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.onRight()")
        
        if self.flagBlockUI: return
        elif self.fPath == '': return
        if isLoadingOneFrame: self.isRunning = False 

        if self.isRunning: # if continuous analysis is running 
            ### FPS update
            self.fps += 1
            if time()-self.last_fps_time >= 1:
                sTxt = wx.FindWindowByName("fps_sTxt", self.panel["tp"])
                sTxt.SetLabel( "FPS: %i"%(self.fps) )
                self.fps = 0
                self.last_fps_time = time()
        
        if self.isRunning and self.vRW.fi >= self.vRW.nFrames-1:
        # continuous running reached the end of frames
            self.onSpace(None) # stop continuous running

        if self.vRW.fi >= self.vRW.nFrames-1: return
        self.vRW.getFrame(-1) # read one frame
        self.proc_img() # process the frame
        
    #-------------------------------------------------------------------

    def onSpace(self, event):
        """ start/stop continuous frame analysis

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("AnimalBehaviourCoderFrame.onSpace()")

        if self.flagBlockUI: return

        if self.fPath == '': return 
        if self.isRunning == False:
            if self.vRW.fi >= self.vRW.nFrames-1: return
            self.isRunning = True
            self.fps = 0
            self.last_fps_time = time()
            self.timer["run"] = wx.CallLater(1, self.onRight)
        else:
            try: # stop timer
                self.timer["run"].Stop() 
                self.timer["run"] = None
            except:
                pass
            sTxt = wx.FindWindowByName("fps_sTxt", self.panel["tp"])
            sTxt.SetLabel('')
            self.isRunning = False # stop continuous analysis
            
    #-------------------------------------------------------------------
    
    def onMLBD_dispImg(self, event):
        if DEBUG: print("AnimalBehaviourCoderFrame.onMLBD_dispImg()")
        if self.fPath == '': return
        mp = event.GetPosition()
        self.isLBPressed = True
        
        ### set base position (to determine head direction and position)
        if self.ratFImgDispImg != None:
            r = 1.0/self.ratFImgDispImg
            self.bPos = (int(mp[0]*r), int(mp[1]*r))
        else:
            self.bPos = (mp[0], mp[1])
        
        self.tmp_img = self.vRW.currFrame.copy()
    
    #-------------------------------------------------------------------

    def onMMove_dispImg(self, event):
        if DEBUG: print("AnimalBehaviourCoderFrame.onMMove_dispImg()")
        if self.fPath == '': return
        if self.isLBPressed == True:
            mp = event.GetPosition()
            if self.ratFImgDispImg != None:
                r = 1.0/self.ratFImgDispImg
                hPos = ( int(mp[0]*r), int(mp[1]*r) )
            else:
                hPos = (mp[0], mp[1])
            mInput = dict(hPosX=hPos[0], hPosY=hPos[1], 
                          bPosX=self.bPos[0], bPosY=self.bPos[1]) 
            ret, frame_arr = self.cv_proc.proc_img(self.tmp_img.copy(), 
                                                   self.animalECase,
                                                   mInput,
                                                   True)
            self.displayAnalyzedImage(frame_arr)
    
    #-------------------------------------------------------------------

    def onMLBU_dispImg(self, event):
        if DEBUG: print("AnimalBehaviourCoderFrame.onMLBU_dispImg()")
        if self.fPath == '': return
        mp = event.GetPosition()
        if self.ratFImgDispImg != None:
            r = 1.0/self.ratFImgDispImg
            hPos = (int(mp[0]*r), int(mp[1]*r))
        else:
            hPos = (mp[0], mp[1])
        self.oData[self.vRW.fi][self.mhpi] = "True"
        self.oData[self.vRW.fi][self.mhdi] = "True"
        mInput = dict(hPosX=hPos[0], hPosY=hPos[1], 
                      bPosX=self.bPos[0], bPosY=self.bPos[1]) 
        self.proc_img(mInput)
        self.isLBPressed = False
        self.bPos = None
        self.tmp_img = None

    #-------------------------------------------------------------------
 
    def onMRBU_dispImg(self, event):
        if DEBUG: print("AnimalBehaviourCoderFrame.onMRBU_dispImg()")
        if self.fPath == '': return
        for ci in range(len(self.dataCols)):
            col = str(self.dataCols[ci])
            if ('PosX' in col) or ('PosY' in col) or \
              (col == 'hD') or (col == 'bD'):
                value = 'D' # deleted data
            elif col in ['mHPos', 'mHD']:
                value = 'True'
            else:
                continue
            self.oData[self.vRW.fi][ci] = value
        self.proc_img()

    #-------------------------------------------------------------------
    
    def applyChangedParam(self):
        """ Apply changed parameters.

        Args: None

        Returns: None
        """ 
        if DEBUG: print("AnimalBehaviourCoderFrame.applyChangedParam()")
        ecp = self.aecParam
        for key in sorted(ecp.keys()):
            valWid = wx.FindWindowByName( '%s_txt'%(key), self.panel["lp"] )
            val = valWid.GetValue().strip()
            currVal = ecp[key]["value"]
            if type(currVal) == list:
                vals = val.split(",")
                if len(currVal) != len(vals):
                    msg = "There should be %i items"%(len(ecp[key]))
                    msg += " for %s."%(key)
                    wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                    return
                else:
                    for i in range(len(vals)):
                        try:
                            if key.startswith("ColorHPos"):
                                vals[i] = float(vals[i])
                            else:
                                vals[i] = int(vals[i])
                        except:
                            msg = "Data type of %s doesn't match."%(key)
                            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                            return
                for i in range(len(vals)): ecp[key]["value"][i] = vals[i]
            else:
                try: val = int(val)
                except:
                    try: val = float(val)
                    except:
                        msg = "%s should be a number."%(key)
                        wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                        return
                self.aecParam[key]["value"] = val
        self.proc_img() # process image
        msg = "Successfully updated."
        wx.MessageBox(msg, "Info", wx.OK|wx.ICON_INFORMATION)
    
    #-------------------------------------------------------------------
    
    def displayAnalyzedImage(self, img):
        """ Display the image, analyzed with cv_proc, in StaticBitmap object

        Args:
            img (numpy.ndarray): Image to display

        Returns:
            None
        """ 
        if DEBUG: print("AnimalBehaviourCoderFrame.displayAnalyzedImage()")
        
        if self.ratFImgDispImg != 1.0:
            img = cv2.resize(img, 
                             (0,0), 
                             fx=self.ratFImgDispImg, 
                             fy=self.ratFImgDispImg)
        
        ### display image 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        wxImg = wx.Image(img.shape[1], img.shape[0])
        wxImg.SetData(img.tostring())
        self.dispImg_sBmp.SetBitmap(wxImg.ConvertToBitmap())
    
    #-------------------------------------------------------------------
    
    def proc_img(self, mInput=None):
        """ Process image with cv_proc module, update resultant data.
        
        Args:
            mInput (None/dict): Manual user input such as 
              mouse click & drag.

        Returns:
            None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.proc_img()")
        if self.fPath == '': return
        
        ### set temporary (for processing the current frame) 
        ###   dictionary to store values
        x = {} # temp. dictionary
        flagMHPos = False 
        for dIdx, dCol in enumerate(self.dataCols):
            if mInput != None and dCol in mInput.keys():
            # manual input is given
                x[dCol] = mInput[dCol]
                flagMHPos = True
            elif self.oData[self.vRW.fi][dIdx] != 'None':
            # already calculated data available
                x[dCol] = self.oData[self.vRW.fi][dIdx]
            else:
                x[dCol] = self.dataInitVal[dIdx]
            ### data from previous frame
            pk = "p_" + dCol
            if self.vRW.fi == 0:
                x[pk] = self.dataInitVal[dIdx] 
            else:
                x[pk] = self.oData[self.vRW.fi-1][dIdx]
                if not x[pk] in ['None', 'D', 'True', 'False']:
                    try: x[pk] = int(x[pk])
                    except: pass
        
        # process 
        ret, frame_arr = self.cv_proc.proc_img(self.vRW.currFrame.copy(), 
                                               self.animalECase,
                                               x,
                                               flagMHPos,
                                               self.dispImgType)
        # display the processed frame 
        self.displayAnalyzedImage(frame_arr)
        
        ### update oData
        for dIdx, dCol in enumerate(self.dataCols):
            self.oData[self.vRW.fi][dIdx] = str(ret[dCol])
        
        if self.flagContManualInput:
            self.oData[self.vRW.fi][self.mhdi] = "True"
        
        if self.oData[self.vRW.fi][self.mhpi] == "True":
        # head position is manually determined via mouse-click on image 
            if len(self.dataGridSelectedCells) > 1:
            # if multiple cells are selected in dataGrid
                for i in range(len(self.dataGridSelectedCells)):
                # going through all selected cells 
                    ri = self.dataGridSelectedCells[i][0] # row index
                    if ri != self.vRW.fi:
                        # copy data of the current frame 
                        #   to other selected frames 
                        self.oData[ri] = list(self.oData[self.vRW.fi]) 
      
        ### update data grid position to make newly calculated data visible 
        self.dataGrid.MakeCellVisible(self.vRW.fi, 0)
        self.dataGrid.SetGridCursor(self.vRW.fi, 0)
         
        if self.isRunning:
            if self.vRW.fi < self.vRW.nFrames-1: # there's more frames to run
                # continue to analyse
                self.timer["run"] = wx.CallLater(1, self.onRight)
    
    #-------------------------------------------------------------------
     
    def initDataWithLoadedVideo(self):
        """ Init. info of loaded video and init/load result data.

        Args: None

        Returns: None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.initDataWithLoadedVideo()") 
        
        ### check bg file
        #bgFile = self.fPath + "_bg.jpg"
        ext = "." + self.fPath.split(".")[-1]
        bgFile = self.fPath.replace(ext, "_bg.jpg")
        if path.isfile(bgFile):
            # load background image
            self.cv_proc.bg = load_img(bgFile, flag='cv')
        
        if self.flagVRec:
            ### start video recorder
            r_video_path = self.fPath.replace(ext, "_ABC")
            if self.vRW.vRecVideoCodec in ['avc1', 'h264']:
                r_video_path += ".mp4"
            elif self.vRW.vRecVideoCodec == 'xvid':
                r_video_path += ".avi"
            w = int(np.ceil(self.vRW.currFrame.shape[1]*self.vRecSzR))
            h = int(np.ceil(self.vRW.currFrame.shape[0]*self.vRecSzR))
            video_fSz = (w, h)
            self.vRW.initWriter(r_video_path, video_fSz) 

        ### init result data (oData)
        result_csv_file = self.fPath + '.csv'
        oData = []
        if path.isfile(result_csv_file):
        # if there's previous result file for this video
            for fi in range(self.vRW.nFrames):
                oData.append(list(self.dataInitVal))
            # load previous CSV data
            oData, endDataIdx = self.loadData(result_csv_file, oData)
        else:
            for fi in range(self.vRW.nFrames):
                oData.append(tuple(self.dataInitVal))
        # set output data as NumPy character array
        self.oData = np.asarray(oData, dtype=self.dataStruct) 
         
        self.onChoice(None, "animalECase_cho") # to init left panel (parameters)

        self.resetDataGrid()
        
        self.session_start_time = time()
        ### set timer for updating the current session running time
        self.timer["sessionTime"] = wx.Timer(self)
        self.Bind(wx.EVT_TIMER,
                  lambda event: self.onTimer(event, "sessionTime"),
                  self.timer["sessionTime"])
        self.timer["sessionTime"].Start(1000)

        if path.isfile(result_csv_file) and endDataIdx > 0:
        # result CSV file exists & 
        # there's, at least, one data (with head direction) exists 
            self.jumpToFrame(endDataIdx) # move to the 1st None value

        btn = wx.FindWindowByName("analyze_btn", self.panel["tp"])
        btn.SetLabel('Stop analysis')
        sTxt = wx.FindWindowByName("fp_sTxt", self.panel["tp"])
        sTxt.SetLabel('%s'%(path.basename(self.fPath)))

        self.proc_img() # process current frame
    
    #-------------------------------------------------------------------
    
    def startStopAnalyzeVideo(self):
        ''' Processing when start or stop analysis
        '''
        if DEBUG: print("AnimalBehaviourCoderFrame.startStopAnalyzeVideo()")
        if self.session_start_time == -1:
        # not in analysis session. start a session
            if self.fPath == '':
                ### select video, if it's not already chosen
                wc = "MP4 files (*.mp4)|*.mp4" 
                wc += "|MOV files (*.mov)|*.mov"
                wc += "|AVI files(*.avi)|*.avi"
                dlg = wx.FileDialog(self, 
                                    "Open video file", 
                                    wildcard=wc, 
                                    style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
                #dlg = wx.DirDialog(self, 
                #                   "Choose directory for analysis", 
                #                   getcwd(), 
                #                   wx.DD_DEFAULT_STYLE|wx.DD_DIR_MUST_EXIST)
                if dlg.ShowModal() == wx.ID_CANCEL: return
                dlg.Destroy()
                self.fPath = dlg.GetPath()
            sTxt = wx.FindWindowByName("fp_sTxt", self.panel["tp"])
            sTxt.SetLabel("Processing... Please wait.")
            self.vRW.initReader(self.fPath) # load video file to analyze
            # calculate ratio to resize to display it in UI
            self.ratFImgDispImg = calcI2DIRatio(self.vRW.currFrame, 
                                                self.dispImg_sBmp_sz)
            self.timer["initData"] = wx.CallLater(10, 
                                                  self.initDataWithLoadedVideo)
            
        else: # in session. stop it.
            dlg = PopupDialog(self, 
                              title="Query", 
                              msg="Save data?", 
                              flagCancelBtn=True)
            rslt = dlg.ShowModal()
            dlg.Destroy()
            if rslt == wx.ID_OK:
                self.onSave(None) # save data
            if self.isRunning:
                self.onSpace(None) # stop continuous running
            if self.flagVRec:
                self.vRW.closeWriter() # stop analysis video recording
            self.cv_proc.bg = None # remove background image
            self.resetDataGrid(flagRemoveOnly=True) # reset data grid
            self.vRW.closeReader() # close video
            ### init
            sTxt = wx.FindWindowByName("ssTime_sTxt", self.panel["tp"])
            sTxt.SetLabel('0:00:00')
            btn = wx.FindWindowByName("analyze_btn", self.panel["tp"])
            btn.SetLabel("Analyze video")
            sTxt = wx.FindWindowByName("fp_sTxt", self.panel["tp"])
            sTxt.SetLabel('')
            self.dispImg_sBmp.SetBitmap(wx.NullBitmap)
            self.session_start_time = -1
            self.fPath = ''
            self.oData = []
   
    #-------------------------------------------------------------------
  
    def loadData(self, result_csv_file, oData):
        """ Load data from CSV file
        """ 
        endDataIdx = -1 
        ### read CSV file and update oData
        f = open(result_csv_file, 'r')
        lines = f.readlines()
        f.close()
        for li in range(1, len(lines)):
            items = [x.strip() for x in lines[li].split(',')]
            if len(items) <= 1: continue

            ### restore parameters
            if items[0] == 'spType':
                self.animalECase = items[1]
                continue
            if items[0] in self.aecParam.keys():
                val = items[1].strip("[]").split("/")
                for vi in range(len(val)):
                    try: val[vi] = int(val[vi])
                    except:
                        try: val[vi] = float(val[vi])
                        except: pass
                if len(val) == 1: val = val[0]
                self.aecParam[items[0]]['value'] = val 
                continue

            ### restore data
            try: fi = int(items[0])
            except: continue
            for ci in range(len(self.dataCols)):
                val = str(items[ci+1])
                oData[fi][ci] = val
                if endDataIdx == -1:
                    if ci == self.hdi and val == 'None':
                    # store frame-index of 'None' value in head direction
                        endDataIdx = copy(fi) 
            oData[fi] = tuple(oData[fi]) # make row as a tuple 
              # for structured numpy array

        if endDataIdx == -1: endDataIdx = fi 
        return (oData, endDataIdx)
    
    #-------------------------------------------------------------------

    def resetDataGrid(self, flagRemoveOnly=False):
        """ Reset dataGrid with data (self.oData)
        
        Args:
            flagRemoveOnly (bool): Remove data without resetting 
                with self.oData

        Returns:
            None
        """ 
        if DEBUG: print("AnimalBehaviourCoderFrame.resetDataGrid()")

        key = "rp"
        pSz = self.pi[key]["sz"] 
        self.gbs[key].Detach(self.dataGrid) # detach grid from gridBagSizer
        self.dataGrid.Destroy() # destroy the grid
        if flagRemoveOnly:
            ### put empty grid and return
            self.dataGrid = Grid(self.panel[key], np.zeros((0,0))) 
            add2gbs(self.gbs[key], self.dataGrid, (0,0), (1,2))
        else:
            ### put a new grid with oData
            self.dataGrid = Grid(self.panel[key],
                                 np.asarray(self.oData), 
                                 size=(pSz[0]-10, pSz[1]-75))
            add2gbs(self.gbs[key], self.dataGrid, (0,0), (1,2))
            ### set column and row labels for dataSheet
            for ci in range(len(self.dataCols)):
                self.dataGrid.SetColLabelValue(ci, self.dataCols[ci])
            #for ri in range(1, self.oData.shape[0]+1):
            #    self.dataGrid.SetRowLabelValue(ri, "%i"%(ri))
            self.dataGrid.AutoSizeColumns() # this could take quite some time,
              # as number of columns increases
        self.gbs[key].Layout()

    #-------------------------------------------------------------------
     
    def onDataGridCellChanged(self, event):
        ''' manual data editing on dataGrid
        '''
        if DEBUG: print("AnimalBehaviourCoderFrame.onDataGridCellChanged()")
        if self.dataGridSelectedCells != []:
            ri = self.dataGrid.GetGridCursorRow()
            ci = self.dataGrid.GetGridCursorCol()
            value = self.oData[ri][ci].lower()
            tmp = None
            try: tmp = int(value)
            except: pass
            if tmp == None: # value is not an integer
                if value == 'true': value = 'True'
                elif value == 'false': value = 'False'
                elif value == 'd': value = 'D'
                elif value == 'none': value = 'None'
            ### update selected cells with entered value
            for i in range(len(self.dataGridSelectedCells)):
                ri = self.dataGridSelectedCells[i][0]
                ci = self.dataGridSelectedCells[i][1]
                self.oData[ri][ci] = str(value)
            self.dataGridSelectedCells = []
     
    #-------------------------------------------------------------------
    
    def onDataGridCellsSelected(self, event):
        ''' store selected multiple cells
        '''
        if DEBUG: print("AnimalBehaviourCoderFrame.onDataGridCellsSelected()")
        if self.dataGrid.GetSelectionBlockTopLeft():
            self.dataGridSelectedCells = []
            tL = self.dataGrid.GetSelectionBlockTopLeft()[0]
            bR = self.dataGrid.GetSelectionBlockBottomRight()[0]
            for ri in range(tL[0], bR[0]+1):
                for ci in range(tL[1], bR[1]+1):
                    self.dataGridSelectedCells.append( (ri,ci) )
    
    #-------------------------------------------------------------------
    
    def callback(self, rData):
        """ call back function after running thread

        Args:
            rData (tuple): Received data from queue at the end of thread running
        
        Returns:
            None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.callbackFunc()")

        if self.diff_FI_TFI> 1: # if moving multiple frames
            # update last_motion_frame
            #   to prevent difference goes over motion detection threshold 
            self.cv_proc.last_motion_frame = self.vRW.currFrame.copy()
        self.proc_img() # process loaded image
        self.flagBlockUI = False
    
    #-------------------------------------------------------------------
    
    def jumpToFrame(self, targetFI=-1):
        """ leap forward to a frame, selected in dataGrid 
        """ 
        if DEBUG: print("AnimalBehaviourCoderFrame.jumpToFrame()")

        if self.fPath == '': return
        if targetFI == -1:
        # if target frame index is NOT given with argument,
            # currently selected row in data grid is the target index
            targetFI = self.dataGrid.GetGridCursorRow()

        if targetFI >= self.vRW.nFrames: targetFI = self.vRW.nFrames-1
         
        # difference between current frame-index and target-frame-index 
        self.diff_FI_TFI= abs(self.vRW.fi - targetFI)
        sTxt = wx.FindWindowByName("navProg_sTxt", self.panel["rp"])
        self.vRW.getFrame(targetFI, self.callback, sTxt)
        self.flagBlockUI = True

    #-------------------------------------------------------------------
    
    def onSave(self, event):
        """ Saving anaylsis result to CSV file.

        Args: event (wx.Event)

        Returns: None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.onSave()")

        nfHPM = 0 # number of frames in which head position is missing
        nfHDM = 0 # number of frames in which head direction is missing
        nfMHD = 0 # number of frames in which head direction is 
          # manually determined
        fp = self.fPath + '.csv'
        fh = open(fp, 'w')

        ### write parameters
        fh.write("Timestamp, %s\n"%(get_time_stamp()))
        fh.write("spType, %s\n"%(self.animalECase))
        for key in sorted(self.aecParam.keys()):
            val = str(self.aecParam[key]['value'])
            if "," in val: val = val.replace(",", "/")
            line = "%s, %s\n"%(key, val)
            fh.write(line)
        fh.write('-----\n')

        ### write column heads 
        line = "frame-index, "
        for col in self.dataCols: line += "%s, "%(col)
        line = line.rstrip(", ") + "\n"
        fh.write(line)
        
        ### write data 
        for fi in range(self.vRW.nFrames):
            line = "%i, "%(fi)
            for ci in range(len(self.dataCols)): line += "%s, "%(str(self.oData[fi][ci]))
            line = line.rstrip(", ") + "\n"
            fh.write(line)
            if self.oData[fi][self.hxi] in ["None", "D"]:
            # head position is missing
                nfHPM += 1 
            if self.oData[fi][self.hdi] in ["None", "D"]:
            # head direction is missing
                nfHDM += 1
            if self.oData[fi][self.mhdi] == "True":
            # head direction is manually determined 
                nfMHD += 1
        fh.write('-----\n')
        fh.write("Number of frames, head position is missing, %i\n"%(nfHPM))
        fh.write("Number of frames, head direction is missing, %i\n"%(nfHDM))
        txt = "Number of frames, head direction is manually determined,"
        txt += " %i\n"%(nfMHD)
        fh.write(txt)
        fh.close()

        msg = 'Saved.\n'
        msg += fp
        wx.MessageBox(msg, "Info", wx.OK|wx.ICON_INFORMATION)

    #-------------------------------------------------------------------

    def onTimer(self, event, flag):
        """ Processing on wx.EVT_TIMER event
        
        Args:
            event (wx.Event)
            flag (str): Key (name) of timer
        
        Returns:
            None
        """
        #if DEBUG: print("AnimalBehaviourCoderFrame.onTimer()")

        if flag == "sessionTime": 
            ### update session running time
            if self.session_start_time != -1:
                e_time = time() - self.session_start_time
                lbl = str(timedelta(seconds=e_time)).split('.')[0]
                sTxt = wx.FindWindowByName("ssTime_sTxt", self.panel["tp"])
                sTxt.SetLabel(lbl)
   
    ''' 
    #-------------------------------------------------------------------
    
    def dataRevision(self):
        """ Data revision with the result CSV file

        Args: None

        Returns: None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.dataRevision()")

        if self.fPath != '': # analysis is ongoing
            msg="Please finish analysis first."
            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
            return
        
        ### open revision dialog
        dlg = ReviseCSV(self, csvFP)
        rslt = dlg.ShowModal()
        
        ### close
        dlg.vRW.closeReader()
        dlg.Destroy()
        self.fPath = ''
    ''' 
    
    #-------------------------------------------------------------------

    def showStatusBarMsg(self, txt, delTime=5000):
        """ Show message on status bar

        Args:
            txt (str): Text to show on status bar
            delTime (int): Duration (in milliseconds) to show the text

        Returns:
            None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.showStatusBarMsg()")

        if self.timers["sb"] != None:
            ### stop status-bar timer
            self.timers["sb"].Stop()
            self.timers["sb"] = None
        
        # show text on status bar 
        self.statusbar.SetStatusText(txt)
        
        ### change status bar color
        if txt == '': bgCol = self.sbBgCol 
        else: bgCol = '#99ee99'
        self.statusbar.SetBackgroundColour(bgCol)

        if txt != '' and delTime != 0:
        # showing message and deletion time was given.
            # schedule to delete the shown message
            self.timers["sb"] = wx.CallLater(delTime,
                                             self.showStatusBarMsg,
                                             '') 

    #-------------------------------------------------------------------

    def onClose(self, event):
        """ Close this frame.
        
        Args: event (wx.Event)
        
        Returns: None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.onClose()")

        stopAllTimers(self.timer)
        rslt = True
        if self.isRunning: # continuous analysis is running
            msg = "Session is running.\n"
            msg += "Unsaved data will be lost. (Stop analysis or"
            msg += " Cmd+S to save.)\nOkay to proceed to exit?"
            dlg = PopupDialog(title="Query", msg=msg, flagCancelBtn=True) 
            rslt = dlg.ShowModal()
            dlg.Destroy()
        if rslt:
            if hasattr(self.vRW, "video_rec") and self.vRW.video_rec != None:
                self.vRW.closeWriter()
            wx.CallLater(500, self.Destroy)
    
    #-------------------------------------------------------------------

#=======================================================================

class TableBase(wx.grid.GridTableBase):
    def __init__(self, data):
        if DEBUG: print("TableBase.__init__()")
        wx.grid.GridTableBase.__init__(self)
        self.data = data
        self.colLabels = list(range(len(data.dtype)))
        # frame indices
        self.rowLabels = [str(x) for x in range(data.shape[0])]
    
    #-------------------------------------------------------------------

    def GetNumberRows(self): 
        #if DEBUG: print("TableBase.GetNumberRows()")
        return self.data.shape[0] 

    #-------------------------------------------------------------------

    def GetNumberCols(self): 
        #if DEBUG: print("TableBase.GetNumberCols()")
        return len(self.data.dtype)

    #-------------------------------------------------------------------

    def GetValue(self, row, col):
        #if DEBUG: print("TableBase.GetValue()")
        return self.data[row][col]

    #-------------------------------------------------------------------

    def SetValue(self, row, col, value):
        #if DEBUG: print("TableBase.SetValue()")
        self.data[row][col] = value
    
    #-------------------------------------------------------------------
    
    def GetRowLabelValue(self, row):
        #if DEBUG: print("TableBase.GetRowLabelValue()")
        return self.rowLabels[row]
    
    #-------------------------------------------------------------------
    
    def SetRowLabelValue(self, row, value):
        #if DEBUG: print("TableBase.SetRowLabelValue()")
        self.rowLabels[row] = value
    
    #-------------------------------------------------------------------
    
    def GetColLabelValue(self, col):
        #if DEBUG: print("TableBase.GetColLabelValue()")
        return self.colLabels[col]
    
    #-------------------------------------------------------------------

    def SetColLabelValue(self, col, value):
        #if DEBUG: print("TableBase.SetColLabelValue()")
        self.colLabels[col] = value

    #-------------------------------------------------------------------

#=======================================================================

class Grid(wx.grid.Grid): 
    def __init__(self, parent, data, size=(100,50)): 
        if DEBUG: print("Grid.__init__()")
        wx.grid.Grid.__init__(self, parent, -1, size=size) 
        self.table = TableBase(data) 
        self.SetTable(self.table, True) 

#=======================================================================

class AnimalBehaviourCoderApp(wx.App):
    def OnInit(self):
        if DEBUG: print("AnimalBehaviourCoderApp.OnInit()")
        self.frame = AnimalBehaviourCoderFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

#=======================================================================

if __name__ == '__main__':
    if len(argv) > 1:
        if argv[1] == '-w': GNU_notice(1)
        elif argv[1] == '-c': GNU_notice(2)
    else:
        GNU_notice(0)
        CWD = getcwd()
        app = AnimalBehaviourCoderApp(redirect = False)
        app.MainLoop()


