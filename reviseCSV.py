# coding: UTF-8

"""
reviseCSV
An open-source software written in Python
  for revising CSV data, produced by pyABC.py 

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
* Left mouse click (in graph): Move to the clicked data frame.
  In selection mode, it will mark start/end point of selection instead. 
* Spacebar: Start/Stop playing video.
* Shift + S: Turn on/off selection mode.
* Left arrow: Go backward for one frame.
* Right arrow: Go forward for one frame.
* Alt + Left arrow: Go backward for width of graph, showin in a screen.
* Alt + Right arrow: Go forwardfor width of graph, showin in a screen.
* Cmd + Left arrow: Go backward to the beginning of data.
* Cmd + Right arrow: Go backward to the end of data. 
* Cmd + U: Undo.
* Cmd + Q: Quit this app.  
"""

import queue
from threading import Thread 
from sys import argv
from os import getcwd, path
from glob import glob
from random import randint
from copy import copy, deepcopy

import cv2
import wx, wx.adv
#from wx.lib.wordwrap import wordwrap
import wx.lib.scrolledpanel as SPanel 
import numpy as np
from scipy.stats import circmean

from videoRW import VideoRW
from fFuncNClasses import GNU_notice, get_time_stamp, getWXFonts
from fFuncNClasses import convt_180_to_360, convt_360_to_180, str2num
from fFuncNClasses import updateFrameSize, add2gbs, receiveDataFromQueue
from fFuncNClasses import stopAllTimers, calc_pt_w_angle_n_dist, calcI2DIRatio

DEBUG = False
VERSION = "0.1.1"

#=======================================================================

class ReviseCSVFrame(wx.Frame):
    """ For revising CSV data, produced by pyABC 

    Args:
        None
     
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self):
        if DEBUG: print("ReviseCSV.__init__()")

        ### init 
        wPos = (0, 25)
        wg = wx.Display(0).GetGeometry()
        wSz = (wg[2], int(wg[3]*0.85))
        wx.Frame.__init__(
              self,
              None,
              -1,
              "[pyABCoder] Revise CSV v.%s"%(VERSION), 
              pos = tuple(wPos),
              size = tuple(wSz),
              style=wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER|wx.MAXIMIZE_BOX),
              )
        self.SetBackgroundColour('#AAAABB')

        ### set app icon
        self.tbIcon = wx.adv.TaskBarIcon(iconType=wx.adv.TBI_DOCK)
        icon = wx.Icon("icon.ico")
        self.tbIcon.SetIcon(icon)
        
        ##### [begin] setting up attributes -----
        self.wSz = wSz
        self.fonts = getWXFonts()
        self.fImgFontScale = 1.0 # font scale for writing string on frame image
        self.fImgFontCol = (0,255,0) # font color for writing string on frame
        self.th = None # thread
        self.q2m = queue.Queue() # queue from thread to main
        self.q2t = queue.Queue() # queue from main to thread 
        self.isRunning = False # continuous playing
        self.flagBlockUI = False # block user input 
        self.ratFImgDispImg = None # ratio to resize frame to display image
        pi = self.setPanelInfo() # set panel info
        self.pi = pi
        self.gbs = {} # for GridBagSizer
        self.panel = {} # panels
        self.timer = {} # timers
        # beginning and end of visible frame indices (in graph)
        self.gVisibleFrameIdx = [0, pi["gp"]["sz"][0]]
        self.isSelectionMode = False
        ### set colors for painting graph
        self.selCol = wx.Colour(0,175,0) # color for selected area
        self.mHLineCol = wx.Colour(127,127,127) # horizontal line 
          # along vertically middle point
        self.dataLnCol = wx.Colour(200,200,200) # data line color
        self.currFICol = wx.Colour(255,175,0) # current frame index color
        self.fontCol = wx.Colour(255,255,255) 
        self.selRange = [-1, -1] # range of selected frames
        self.gMarker = [] # markers on graph
        self.gMCol = [] # colors for each marker 
        self.gMMax = 100 # maximum numbers of marker
        self.csvFP = "" # CSV file path
        self.videoFP = "" # video file path
        self.vRW = None # module for reading/writing video file
        self.aecParam = {} # parameters in CSV data
        self.dataCols = [] # column names of CSV data
        self.oData = [] # data from CSV file
        self.endDataIdx = -1 # row index where all data is 'None', 
          # or simply end row index of data
        self.gFI_onMP = -1 # frame-index on graph where mouse point was on 
        ##### [end] setting up attributes -----
        
        
        updateFrameSize(self, wSz)
        
        ### create panels
        for k in pi.keys():
            self.panel[k] = SPanel.ScrolledPanel(self, 
                                                 pos=pi[k]["pos"],
                                                 size=pi[k]["sz"],
                                                 style=pi[k]["style"])
            self.panel[k].SetBackgroundColour(pi[k]["bgCol"])

        ##### [begin] set up top panel interface -----
        vlSz = (-1, 20) # size of vertical line separator
        self.gbs["tp"] = wx.GridBagSizer(0,0)
        row = 0; col = 0
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="Open CSV",
                        name="openCSV_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["tp"], btn, (row,col), (1,2))
        col += 2 
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="Save CSV",
                        name="save_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["tp"], btn, (row,col), (1,2))
        col += 2 
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="Save video",
                        name="saveVid_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["tp"], btn, (row,col), (1,2))
        col += 2 
        add2gbs(self.gbs["tp"],
                wx.StaticLine(self.panel["tp"],
                              -1,
                              size=vlSz,
                              style=wx.LI_VERTICAL),
                (row,col),
                (1,1)) # vertical line separator
        col += 1
        btn = wx.Button(self.panel['tp'], 
                                  wx.ID_OK, 
                                  label='Quit revision',
                                  name="quit_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["tp"], btn, (row,col), (1,2))
        col += 2 
        sTxt = wx.StaticText(self.panel['tp'], 
                             -1,
                             name="csvFP_sTxt",
                             label="")
        sTxt.SetForegroundColour('#ffffff')
        add2gbs(self.gbs["tp"], sTxt, (row,col), (1,10))
        row += 1
        col = 0 
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="|<",
                        name="moveToBegin_btn",
                        size=(30,-1))
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="<<",
                        name="moveBackFur_btn",
                        size=(30,-1))
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label=">>",
                        name="moveForFur_btn",
                        size=(30,-1))
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label=">|",
                        name="moveToEnd_btn",
                        size=(30,-1))
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1 
        txt = wx.TextCtrl(self.panel['tp'], 
                          -1, 
                          name="fIdx_txt", 
                          value='index', 
                          size=(50,-1))
        add2gbs(self.gbs["tp"], txt, (row,col), (1,1)) 
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="go",
                        name="moveToFIdx_btn",
                        size=(40,-1))
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1 
        add2gbs(self.gbs["tp"],
                wx.StaticLine(self.panel["tp"],
                              -1,
                              size=vlSz,
                              style=wx.LI_VERTICAL),
                (row,col),
                (1,1)) # vertical line separator
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="SelectionMode:OFF",
                        name="selMode_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,3))
        col += 3 
        add2gbs(self.gbs["tp"],
                wx.StaticLine(self.panel["tp"],
                              -1,
                              size=vlSz,
                              style=wx.LI_VERTICAL),
                (row,col),
                (1,1)) # vertical line separator
        col += 1
        txt = wx.TextCtrl(self.panel['tp'], 
                          -1, 
                          name="hdVal_txt", 
                          value='', 
                          size=(50,-1))
        add2gbs(self.gbs["tp"], txt, (row,col), (1,1))
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="Set",
                        name="hdSet_btn",
                        size=(50,-1))
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="+",
                        name="hdPlus_btn",
                        size=(30,-1))
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="-",
                        name="hdMinus_btn",
                        size=(30,-1))
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1 
        add2gbs(self.gbs["tp"],
                wx.StaticLine(self.panel["tp"],
                              -1,
                              size=vlSz,
                              style=wx.LI_VERTICAL),
                (row,col),
                (1,1)) # vertical line separator
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="DEL",
                        name="hdDel_btn",
                        size=(40,-1))
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1 
        add2gbs(self.gbs["tp"],
                wx.StaticLine(self.panel["tp"],
                              -1,
                              size=vlSz,
                              style=wx.LI_VERTICAL),
                (row,col),
                (1,1)) # vertical line separator
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="Linear interpolation",
                        name="hdLinInt_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1 
        add2gbs(self.gbs["tp"],
                wx.StaticLine(self.panel["tp"],
                              -1,
                              size=vlSz,
                              style=wx.LI_VERTICAL),
                (row,col),
                (1,1)) # vertical line separator
        col += 1
        cho = wx.Choice(self.panel['tp'], 
                        -1, 
                        choices=[str(x) for x in range(3,31,2)], 
                        name="refL_cho",
                        size=(50,-1))
        cho.SetSelection(1)
        add2gbs(self.gbs["tp"], cho, (row,col), (1,1))
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="Smooth",
                        name="smooth_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1 
        add2gbs(self.gbs["tp"],
                wx.StaticLine(self.panel["tp"],
                              -1,
                              size=vlSz,
                              style=wx.LI_VERTICAL),
                (row,col),
                (1,1)) # vertical line separator
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="Clear markers",
                        name="clearMarker_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1)) 
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label="Undo",
                        name="undo_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown) 
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))  
        self.panel["tp"].SetSizer(self.gbs["tp"])
        self.gbs["tp"].Layout()
        self.panel["tp"].SetupScrolling()
        ##### [end] set up top panel interface -----

        # set up middle panel (for displaying frame image)
        self.dispImg_sBmp = wx.StaticBitmap(self.panel['mp'], 
                                            -1, 
                                            wx.NullBitmap, 
                                            (0,0), 
                                            pi['mp']['sz'] )
        
        ### set up graph-panel
        self.panel["gp"].Bind(wx.EVT_PAINT, self.onPaint)
        self.panel["gp"].Bind(wx.EVT_LEFT_UP, self.onClickGraph)
        self.panel["gp"].Bind(wx.EVT_RIGHT_UP, self.onMouseRClick)
        self.panel["gp"].Bind(wx.EVT_MOTION, self.onMouseMove)

        ##### [begin] set up bottom panel interface -----
        self.gbs["bp"] = wx.GridBagSizer(0,0)
        row = 0; col = 0
        self.bp_sTxt = wx.StaticText(self.panel['bp'], -1, label="")
        self.bp_sTxt.SetForegroundColour('#cccccc')
        add2gbs(self.gbs["bp"], self.bp_sTxt, (row,col), (1,1))
        self.panel["bp"].SetSizer(self.gbs["bp"])
        self.gbs["bp"].Layout()
        #self.panel["bp"].SetupScrolling()
        ##### [end] set up bottom panel interface -----

        ### keyboard binding
        exit_btnId = wx.NewIdRef(count=1)
        space_btnId = wx.NewIdRef(count=1) # for continuous playing 
        selection_btnId = wx.NewIdRef(count=1) # for selection mode on/off
        undo_btnId = wx.NewIdRef(count=1) # for undo
        backFur_btnId = wx.NewIdRef(count=1)
        forFur_btnId = wx.NewIdRef(count=1)
        backBegin_btnId = wx.NewIdRef(count=1)
        forEnd_btnId = wx.NewIdRef(count=1)
        self.Bind(wx.EVT_MENU, self.onClose, id = exit_btnId)
        self.Bind(wx.EVT_MENU, self.onSpace, id = space_btnId)
        self.Bind(wx.EVT_MENU, self.selectionModeOnOff, id = selection_btnId)
        self.Bind(wx.EVT_MENU, self.undo, id = undo_btnId)
        self.Bind(wx.EVT_MENU, 
                  lambda event: self.moveFrame(event, 'backFur'), 
                  id=backFur_btnId)
        self.Bind(wx.EVT_MENU, 
                  lambda event: self.moveFrame(event, 'forFur'), 
                  id=forFur_btnId)
        self.Bind(wx.EVT_MENU, 
                  lambda event: self.moveFrame(event, 'backBegin'), 
                  id=backBegin_btnId)
        self.Bind(wx.EVT_MENU, 
                  lambda event: self.moveFrame(event, 'forEnd'), 
                  id=forEnd_btnId)
        accel_tbl = wx.AcceleratorTable([
                            (wx.ACCEL_CMD,  ord('Q'), exit_btnId ),
                            (wx.ACCEL_NORMAL, wx.WXK_SPACE, space_btnId),
                            (wx.ACCEL_SHIFT, ord('S'), selection_btnId),
                            (wx.ACCEL_CMD, ord('U'), undo_btnId),
                            (wx.ACCEL_ALT,  wx.WXK_LEFT, backFur_btnId), 
                            (wx.ACCEL_ALT,  wx.WXK_RIGHT, forFur_btnId),
                            (wx.ACCEL_CTRL,  wx.WXK_LEFT, backBegin_btnId), 
                            (wx.ACCEL_CTRL,  wx.WXK_RIGHT, forEnd_btnId), 
                                        ])
        self.SetAcceleratorTable(accel_tbl)
         
        self.openCSVFile()
     
    #-------------------------------------------------------------------
   
    def setPanelInfo(self):
        """ Set up panel information.
        
        Args:
            None
        
        Returns:
            pi (dict): Panel information.
        """
        if DEBUG: print("ReviseCSV.setPanelInfo()")

        wSz = self.wSz
        pi = {} # information of panels
        bpH = 30 # bottom panel height
        # top panel for major buttons
        pi["tp"] = dict(pos=(0, 0), 
                        sz=(wSz[0], 75), 
                        bgCol="#666666", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        # middle panel for showing frame image
        pi["mp"] = dict(pos=(0, pi['tp']['sz'][1]), 
                        sz=(wSz[0], (wSz[1]-pi['tp']['sz'][1]-bpH)/2), 
                        bgCol="#333333", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        # panel for showing graph
        pi["gp"] = dict(pos=(0, pi['tp']['sz'][1]+pi['mp']['sz'][1]), 
                        sz=(wSz[0], pi['mp']['sz'][1]), 
                        bgCol="#000000", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        tpSz = pi["tp"]["sz"]
        mpSz = pi["mp"]["sz"]
        gpSz = pi["gp"]["sz"]
        # bottom panel for short info
        pi["bp"] = dict(pos=(0, tpSz[1]+mpSz[1]+gpSz[1]), 
                        sz=(wSz[0], bpH), 
                        bgCol="#333333", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        return pi

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
        if DEBUG: print("ReviseCSV.onButtonPressDown()")

        if objName == '':
            obj = event.GetEventObject()
            objName = obj.GetName()
        else:
            obj = wx.FindWindowByName(objName, self.panel["tp"])

        if objName == "save_btn": self.save()
        
        if self.flagBlockUI: return
        if not obj.IsEnabled(): return

        self.playSnd("leftClick")

        if objName == "quit_btn": self.onClose(None)
        elif objName == "openCSV_btn": self.openCSVFile()
        elif objName == "saveVid_btn":
            self.jumpToFrame(0)
            self.timer["saveVideo"] = wx.CallLater(10, self.saveVideo)
        elif objName == "moveToBegin_btn": self.moveFrame(None, 'backBegin')
        elif objName == "moveBackFur_btn": self.moveFrame(None, 'backFur') 
        elif objName == "moveForFur_btn": self.moveFrame(None, 'forFur')
        elif objName == "moveToEnd_btn": self.moveFrame(None, 'forEnd')
        elif objName == "moveToFIdx_btn": self.moveFrame(None, 'fIdx') 
        elif objName == "selMode_btn": self.selectionModeOnOff(None)
        elif objName == "hdSet_btn": self.changeHDVal('set')
        elif objName == "hdPlus_btn": self.changeHDVal('plus')
        elif objName == "hdMinus_btn": self.changeHDVal('minus')
        elif objName == "hdDel_btn": self.changeHDVal('delete')
        elif objName == "hdLinInt_btn": self.changeHDVal('linearInterpolation')
        elif objName == "smooth_btn": self.changeHDVal('smooth')
        elif objName == "clearMarker_btn": self.clearMarkers()
        elif objName == "undo_btn": self.undo(None)

    #-------------------------------------------------------------------
    
    def onClose(self, event):
        """ Close this frame. 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("ReviseCSV.onClose()")

        stopAllTimers(self.timer)
        if hasattr(self.vRW, "video_rec") and self.vRW.video_rec != None:
            self.vRW.closeWriter()
        wx.CallLater(500, self.Destroy)
    
    #-------------------------------------------------------------------

    def openCSVFile(self):
        """ Open data CSV file. 
        
        Args:
            None
        
        Returns:
            None 
        """
        if DEBUG: print("ReviseCSV.openCSVFile()")

        ### choose result CSV file 
        wc = 'CSV files (*.csv)|*.csv' 
        dlg = wx.FileDialog(self, 
                            "Open CSV file", 
                            wildcard=wc, 
                            style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_CANCEL: return
        dlg.Destroy()
        csvFP = dlg.GetPath()

        ##### [begin] validate chosen CSV ---
        fh = open(csvFP, 'r')
        lines = fh.readlines()
        fh.close()
        ###
        flagFI = False; fIdx = None
        for line in lines:
            items = [ x.strip() for x in line.split(',') ]
            if line.startswith("frame-index"):
                flagFI = True
                continue
            if flagFI:
            # line with column titles already found
                fIdx = str2num(items[0], 'int') # frame-index of 1st data line
                break
        if flagFI == False or fIdx == None:
        # there should be column title line and 
        #   there was, at least, one integer value for frame-index column
            msg = "The chosen CSV file is not a valid result file"
            msg += " for the program."
            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
            return
        ###
        videoFP = csvFP.replace(".csv", "")
        self.videoFP = videoFP # video file path
        if path.isfile(videoFP) == False:
            fn = path.basename(videoFP)
            msg = "Video data folder '%s' should exist"%(fn)
            msg += " where the chosen CSV file is."
            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
            return
        ##### [end] validate chosen CSV ---
        self.csvFP = csvFP
        self.videoFP = videoFP
        sTxt = wx.FindWindowByName("csvFP_sTxt", self.panel["tp"])
        sTxt.SetLabel(path.basename(csvFP))
       
        d = self.loadData(csvFP) # load data
        self.aecParam, self.dataCols, self.oData, self.endDataIdx = d
        self.vRW = VideoRW(self) # for reading/writing video file
        self.vRW.initReader(self.videoFP) # init video file
        self.backupOData = None # backup data

        ### store some data indices
        ### (currently, the app is only for head direction)
        self.hdi = self.dataCols.index("hD")
        self.hxi = self.dataCols.index("hPosX")
        self.hyi = self.dataCols.index("hPosY")
        self.bxi = self.dataCols.index("bPosX")
        self.byi = self.dataCols.index("bPosY")
        self.mhdi = self.dataCols.index("mHD")
        self.mhpi = self.dataCols.index("mHPos")

        self.play() # for processing the 1st frame

    #-------------------------------------------------------------------
  
    def loadData(self, csvFP):
        """ Load data from CSV file

        Args:
            csvFP (str): File path of result CSV.

        Returns:
            aecParam (dict): Parameter values from CSV.
            dataCols: (list): Data columns.
            oData (list): Output data.
            endDataIdx (int): Row index of end of data.
        """ 
        if DEBUG: print("ReviseCSV.loadData()")

        oData = []
        dataCols = []
        aecParam = {}
        endDataIdx = None
        ### read CSV file and update oData
        f = open(csvFP, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            items = [x.strip() for x in line.split(',')]
            if len(items) <= 1: continue

            ### store data columns 
            if line.startswith("frame-index"):
                for ii in range(len(items)):
                    dataCols.append(items[ii])
           
            if dataCols == []:
                ### store parameters
                val = items[1].strip("[]").split("/")
                for vi in range(len(val)):
                    val[vi] = str2num(val[vi])
                if len(val) == 1: val = val[0]
                aecParam[items[0]] = dict(value = val) 
                continue
            
            ### store data
            try: fi = int(items[0])
            except: continue
            flagAllNone = True # whether data in all columns were 'None'
            oDataRow = []
            for ci in range(len(dataCols)):
                val = str(items[ci])
                oDataRow.append(val)
                if dataCols[ci] != "frame-index" and val != "None":
                    flagAllNone = False
            if flagAllNone: endDataIdx = copy(fi-1) # set end data index
            oData.append(oDataRow)
        if endDataIdx == None: endDataIdx = fi
        
        return (aecParam, dataCols, oData, endDataIdx)
    
    #-------------------------------------------------------------------
    
    def onPaint(self, event):
        """ painting graph

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if DEBUG: print("ReviseCSV.onPaint()")

        if self.csvFP == "": return

        gIdx = self.gVisibleFrameIdx
        if gIdx[0] == -1 or gIdx[1] == -1: return

        event.Skip()
        dc = wx.PaintDC(self.panel['gp'])

        ### set bg colors
        if self.isSelectionMode: bgCol = wx.Colour(50,50,50) # background color
        else: bgCol = wx.Colour(0,0,0) 
        
        dc.SetBackground(wx.Brush(bgCol))
        dc.Clear()

        gSz = self.pi['gp']['sz']
        vMid = gSz[1]/2 # vertical middle 

        dc.SetPen(wx.Pen(self.mHLineCol, 1))
        dc.DrawLine(0, vMid, gSz[0], vMid) # horizontal line 
          # along vertically middle point

        ### draw selection area
        sr = list(self.selRange)
        sr[0] = max(0, sr[0]-self.gVisibleFrameIdx[0])
        if sr.count(-1) == 0: # range selection complete
            sr[1] -= self.gVisibleFrameIdx[0]
            if sr[1] >= 0:
                dc.SetPen(wx.Pen(self.selCol, 0))
                dc.SetBrush(wx.Brush(self.selCol))
                dc.DrawRectangle(sr[0], 0, sr[1]-sr[0], gSz[1])
        
        if self.isSelectionMode and self.gFI_onMP != -1:
        # if it's in selection mode
            ### draw a line with selection color, 
            ### where mouse position is currently on.
            li = self.gFI_onMP - gIdx[0]
            dc.SetPen(wx.Pen(self.selCol, 1))
            dc.DrawLine(li, 0, li, gSz[1])

        ### draw data (head direction) lines
        currFrameX = None
        dc.SetPen(wx.Pen(self.dataLnCol, 1))
        for li in range(gIdx[1]-gIdx[0]+1):
            idx = gIdx[0] + li
            if idx >= self.vRW.nFrames: break
            hD = self.oData[idx][self.hdi]
            hD = str2num(hD, 'int')
            if type(hD) == int:
                hdLen = int(hD / 180.0 * vMid) * -1 # head direction range is 
                  # -180 ~ 180
                if idx == self.vRW.fi: # current frame index
                    currFrameX = int(li)
                    dc.SetPen(wx.Pen(self.currFICol, 1))
                    dc.DrawLine(li, 0, li, gSz[1])
                    dc.SetPen(wx.Pen(self.dataLnCol, 1))
                else:
                    dc.DrawLine(li, vMid, li, vMid+hdLen)

                if idx in self.gMarker: # this index is in marker list
                    i = self.gMarker.index(idx)
                    ### draw a marker
                    dc.SetPen(wx.Pen(self.gMCol[i], 2))
                    if hD > 0: 
                        y = max(0, vMid+hdLen-5)
                        dc.DrawLine(li, y+5, li+5, y)
                        dc.DrawLine(li, y+5, li, y)
                    else:
                        y = min(gSz[1], vMid+hdLen+5)
                        dc.DrawLine(li, y-5, li+5, y)
                        dc.DrawLine(li, y-5, li, y)
                    dc.SetPen(wx.Pen(self.dataLnCol, 1))
        
        if sr.count(-1) == 1:
        # if only a start point for selection is chosen 
            ### draw it as a line
            dc.SetPen(wx.Pen(self.selCol, 1))
            dc.DrawLine(sr[0], 0, sr[0], gSz[1]) 

        ### draw status message for the current frame
        if currFrameX != None:
            status_msg = "%i/ %i, %s"%(self.vRW.fi, 
                                       self.vRW.nFrames-1, 
                                       self.oData[self.vRW.fi][self.hdi])
            dc.SetFont(self.fonts[2])
            dc.SetTextForeground(self.fontCol)
            dc.DrawText(status_msg, currFrameX+1, 5) 

    #-------------------------------------------------------------------
    
    def selectionModeOnOff(self, event):
        """ Turn on/off selection mode

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("ReviseCSV.selectionModeOnOff()")

        self.selRange = [-1, -1]
        selMode_btn = wx.FindWindowByName("selMode_btn", self.panel["tp"])
        if self.isSelectionMode:
            self.isSelectionMode = False
            selMode_btn.SetLabel("SelectionMode:OFF")
        else:
            self.isSelectionMode = True 
            selMode_btn.SetLabel("SelectionMode:ON")
        self.panel["gp"].Refresh() # draw graph
    
    #-------------------------------------------------------------------
   
    def changeHDVal(self, flag):
        """ change head direction value(s)

        Args:
            flag (str): Operation type to change head direction values.

        Returns:
            None 
        """ 
        if DEBUG: print("ReviseCSV.changeHDVal()")

        ### get frame index range to apply
        if not -1 in self.selRange:
            fIdx = range(self.selRange[0], self.selRange[1]+1)
        elif flag == 'smooth':
            if self.endDataIdx > 0: fIdx = range(0, self.endDataIdx)
            else: fIdx = range(0, self.vRW.nFrames-1)
        else:
            fIdx = [copy(self.vRW.fi)]
        p = dict(fIdx=fIdx) # parameters

        ### get necessary parameters ready and validation
        if flag == 'smooth':
            cho = wx.FindWindowByName("refL_cho", self.panel["tp"])
            # number of frames to refer backward or forward 
            # (from -refL to +refL of current frame)
            refL = int(cho.GetString(cho.GetSelection()))
            p["refL"] = refL
        
        elif flag == 'linearInterpolation':
        # linear interpolation
            if len(fIdx) == 1:
                msg = "Select range to apply linear interpolation."
                wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                return
            bHD = self.oData[fIdx[0]][self.hdi]
            eHD = self.oData[fIdx[-1]][self.hdi]
            try:
                bHD = int(bHD)
                eHD = int(eHD)
            except:
                msg =  "First and last value in the selected range should be"
                msg += " integers for linear interpolation."
                wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                return
            hDDiff = eHD - bHD
            if (eHD/abs(eHD)) != (bHD/abs(bHD)): # sign doesn't match
                ### for changing over zero or 180 line 
                ### (e.g.: 30 -> -30, -170 -> 170, etc..)
                hDDiff = (180-abs(eHD)) + (180-abs(bHD))
                if hDDiff > 180: hDDiff = 360 - hDDiff
                if (0 <= bHD < hDDiff) or (bHD < 0 and bHD < -hDDiff):
                    hDDiff = -hDDiff
            fLen = len(fIdx)
            p["bHD"] = bHD
            p["eHD"] = eHD
            p["hDDiff"] = hDDiff
            p["fLen"] = fLen
        
        elif flag in ['set', 'plus', 'minus']:
        # set, plus or minus
            ### get input value in textCtrl
            txt = wx.FindWindowByName("hdVal_txt", self.panel["tp"])
            inputVal = txt.GetValue()
            try: inputVal = int(inputVal)
            except:
                msg = "'%s' in the textbox is not an integer."%(inputVal)
                wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                return
            if inputVal < -180 or inputVal > 180:
                msg = "'%s' in the textbox"%(inputVal)
                msg += " should be in range of -180~180."
                wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                return
            if flag == 'minus': inputVal = -inputVal
            p["inputVal"] = inputVal
        
        self.backupOData = deepcopy(self.oData) # back up data
        if self.isSelectionMode:
            self.selectionModeOnOff(None) # turn off selection-mode
        
        if len(fIdx) > 1000:
            self.flagBlockUI = True 
            ### set timer for updating current progress 
            self.timer["changeHDVal"] = wx.Timer(self)
            self.Bind(wx.EVT_TIMER,
                      lambda event: self.onTimer(event, "changeHDVal"),
                      self.timer["changeHDVal"])
            self.timer["changeHDVal"].Start(10) 
            ### start thread to write 
            self.th = Thread(target=self.runChangeHDVal, 
                             args=(p, self.oData, flag, self.q2m, True,))
            wx.CallLater(20, self.th.start)
        else:
            self.oData = self.runChangeHDVal(p, 
                                             self.oData, 
                                             flag, 
                                             isThread=False) # run 
            self.panel["gp"].Refresh() # draw graph
            self.displayFrameImage(self.vRW.currFrame) # show current frame

    #-------------------------------------------------------------------
   
    def runChangeHDVal(self, p, oData, flag, q2m=None, isThread=True):
        """ Change head direction value(s) procedure.
        Separated for running it as a thread.

        Args:
            p (dict): Parameters for operation.
            oData (dict): Output data.
            flag (str): Operation type to change head direction values.
            q2m (queue.Queue): Queue to send data to main thread.
            isThread (bool): Whether running this function as a thread.

        Returns:
            oData (dict): Changed data. 
        """ 
        if DEBUG: print("ReviseCSV.runChangeHDVal()")
        
        ### calculate values
        modifiedFIs = []
        newHDs = []
        for i, fi in enumerate(p["fIdx"]):
            if isThread:
                msg = "calculating.. %i/ %i"%(fi, p["fIdx"][-1])
                q2m.put((msg,), True, None)
            
            if flag == 'smooth': # smooth data line, 
              # referring +/- several frames around the current frame
                if fi < p["refL"] or fi > self.vRW.nFrames-1-p["refL"]: continue
                refHDs = []
                for _fi in range(fi-p["refL"], fi+p["refL"]+1):
                    _hd = oData[_fi][self.hdi]
                    _hd = str2num(_hd, 'int')
                    if type(_hd) == int:
                        # store reference head direction 
                        #   (radian angle in 360 degree system)
                        refHDs.append(np.deg2rad(convt_180_to_360(_hd)))
                if refHDs != []:
                    if fi >= p["refL"]:
                        modifiedFIs.append(fi)
                        cm = np.rad2deg(circmean(refHDs)) # mean value of
                          # reference frames 
                          # head directions -> convert to degree from radian
                        newHDs.append(convt_360_to_180(int(cm)))

            elif flag == 'linearInterpolation':
                if fi == 0 or fi == p["fLen"]-1: continue
                newHD = int(p["bHD"] + p["hDDiff"]*(float(i)/p["fLen"]))
                if newHD > 180: newHD = -(180-newHD%180)
                elif newHD < -180: newHD = 180-(abs(newHD)-180)
                modifiedFIs.append(fi)
                newHDs.append(newHD)

            else: # set, plus or minus
                oldHD = oData[fi][self.hdi]
                oldHD = str2num(oldHD, 'int')
                if flag == 'set':
                    modifiedFIs.append(fi)
                    newHDs.append(p["inputVal"])
                elif flag == 'delete':
                    modifiedFIs.append(fi)
                    newHDs.append('D')
                else:
                    if type(oldHD) == str: # 'None' or 'D'
                        oData[fi][self.hdi] = str(oldHD)
                    else:
                        newHD = oldHD + p["inputVal"]
                        newHD = min(max(-180, newHD), 180) # range is -180~180
                        modifiedFIs.append(fi)
                        newHDs.append(newHD) 

        ### change data 
        for i in range(len(modifiedFIs)):
            fi = modifiedFIs[i]
            if isThread:
                msg = "updating data.. %i/ %i"%(fi, modifiedFIs[-1])
                q2m.put((msg,), True, None)
            
            newHD = newHDs[i]
            _d = oData[fi]
            if type(newHD) == int:
                bPosX = oData[fi][self.bxi]
                bPosY = oData[fi][self.byi]
                if not bPosX in ['None', 'D']:
                    # re-calculate hPos with new head direction value
                    hPos = calc_pt_w_angle_n_dist(
                                            newHD, 
                                            self.aecParam["hdLineLen"]["value"],
                                            int(bPosX),
                                            int(bPosY),
                                            True
                                            )
                    oData[fi][self.hxi] = str(hPos[0])
                    oData[fi][self.hyi] = str(hPos[1])
            oData[fi][self.hdi] = str(newHD)
            oData[fi][self.mhdi] = "True"
            if flag == 'delete':
                oData[fi][self.hxi] = "D" 
                oData[fi][self.hyi] = "D" 
                oData[fi][self.mhpi] = "True"
                oData[fi][self.bxi] = "D" 
                oData[fi][self.byi] = "D"

        if isThread: q2m.put(("", oData), True, None)
        else: return oData
    
    #-------------------------------------------------------------------

    def onTimer(self, event, flag):
        """ Processing on wx.EVT_TIMER event
        
        Args:
            event (wx.Event)
            flag (str): Key (name) of timer
        
        Returns:
            None
        """
        #if DEBUG: print("VideoRW.onTimer()") 

        ### receive (last) data from queue
        rData = None
        while True: 
            ret = receiveDataFromQueue(self.q2m)
            if ret == None: break
            rData = ret # store received data
        if rData == None: return
        
        if flag == "changeHDVal":
            if len(rData) == 1:
                self.bp_sTxt.SetLabel(rData[0])
            elif len(rData) == 2:
            # reached end of process
                self.bp_sTxt.SetLabel(rData[0])
                self.oData = rData[1]
                self.panel["gp"].Refresh() # draw graph
                self.displayFrameImage(self.vRW.currFrame) # show current frame
                self.flagBlockUI = False 
    
    #-------------------------------------------------------------------
    
    def onClickGraph(self, event):
        """ Processing when user clicked graph

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("ReviseCSV.onClickGraph()")

        if self.flagBlockUI: return

        self.playSnd("leftClick") 
        
        mp = event.GetPosition()
        idx = self.gVisibleFrameIdx[0] + mp[0] - 1
        if idx < 0: idx = 0
        elif idx >= len(self.oData): idx = len(self.oData)-1 
        if self.isSelectionMode:
            if self.selRange.count(-1) == 1:
                self.selRange = [min(self.selRange[0], idx),
                                 max(self.selRange[0], idx)]
            else:
                self.selRange = [idx, -1]
            self.panel["gp"].Refresh() # draw graph
        else:
            self.jumpToFrame(idx) 
   
    #-------------------------------------------------------------------
    
    def onMouseMove(self, event):
        """ Mouse pointer moving on graph area
        Show some info

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("ReviseCSV.onMouseMove()")

        if self.flagBlockUI: return

        mp = event.GetPosition()
        idx = self.gVisibleFrameIdx[0] + mp[0] - 1
        if idx < 0 or idx >= len(self.oData): return
               
        ## write current frame info
        d = self.oData[idx]
        if idx in self.gMarker: msg = "[MARKED] "
        else: msg = ""
        msg += "Frame-index: %i, %s"%(idx, d[self.hdi])
        self.bp_sTxt.SetLabel(msg)

        if self.isSelectionMode:
            # store frame index in graph, where mouse pointer is currently on
            self.gFI_onMP = idx
            # draw graph
            self.panel["gp"].Refresh()
    
    #-------------------------------------------------------------------
    
    def onMouseRClick(self, event):
        """ Mouse right click on graph area.

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("ReviseCSV.onMouseRClick()")
        
        if self.flagBlockUI: return

        if len(self.gMarker) == self.gMMax:
        # markers have already max number of markers
            return

        mp = event.GetPosition()
        idx = self.gVisibleFrameIdx[0] + mp[0] - 1
        if idx in self.gMarker: # already in marker list, remove
            i = self.gMarker.index(idx)
            self.gMarker.pop(i)
            self.gMCol.pop(i)
        else:
            self.gMarker.append( idx ) # add marker
            col = wx.Colour(randint(100,255), 
                            randint(100,255), 
                            randint(0,100))
            self.gMCol.append(col) # marker color
        self.panel["gp"].Refresh() # re-draw graph

    #-------------------------------------------------------------------
    
    def clearMarkers(self): 
        """ Clear all markers on graph
        
        Args: None

        Returns: None
        """
        if DEBUG: print("ReviseCSV.clearMarkers()")
        self.gMarker = []
        self.gMCol = []
        self.panel["gp"].Refresh() # re-draw graph
    
    #-------------------------------------------------------------------
    
    def undo(self, event):
        """ Undo last change of head direction data 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("ReviseCSV.undo()")

        if self.backupOData is None: return
        tmp = deepcopy(self.oData)
        self.oData = deepcopy(self.backupOData)
        self.backupOData = tmp
        self.panel["gp"].Refresh() # re-draw graph
    
    #-------------------------------------------------------------------
    
    def moveFrame(self, event, flag):
        """ Move to certain frame, depending on 'flag'.
        
        Args:
            event (wx.Event)
            flag (str): Indicator for moving direction and distance.

        Returns:
            None
        """ 
        if DEBUG: print("ReviseCSV.moveFrame()")

        gw = self.pi['gp']['sz'][0] # graph width 
        if flag == 'backBegin':
            self.jumpToFrame(0)
        elif flag == 'forEnd':
            self.jumpToFrame(self.endDataIdx)
        elif flag == 'backFur':
            self.jumpToFrame(max(0, self.vRW.fi-gw))
        elif flag == 'forFur':
            self.jumpToFrame(min(self.endDataIdx, self.vRW.fi+gw))
        elif flag == 'fIdx':
            txt = wx.FindWindowByName("fIdx_txt", self.panel["tp"])
            fi = str2num(txt.GetValue(), "int")
            if type(fi) == int and fi >= 0 and fi <= self.endDataIdx:
                self.jumpToFrame(fi)
            txt.SetValue("")
    
    #-------------------------------------------------------------------
    
    def jumpToFrame(self, targetFI=-1):
        """ Jump to a frame with idx
        
        Args:
            targetFI (int): Index number to move to. (-1 means next frame)
        
        Returns:
            None
        """
        if DEBUG: print("ReviseCSV.jumpToFrame()")

        gpSz = self.pi["gp"]["sz"]
        if targetFI == -1:
            if self.vRW.fi >= self.vRW.nFrames: return
            self.gVisibleFrameIdx[0] = max(0, int(self.vRW.fi+1-gpSz[0]/2))
        else:
            self.gVisibleFrameIdx[0] = max(0, int(targetFI-gpSz[0]/2))
        self.gVisibleFrameIdx[1] = self.gVisibleFrameIdx[0] + gpSz[0]
        if targetFI == -1:
            self.vRW.getFrame(-1)
            self.displayFrameImage(self.vRW.currFrame) # show current frame
            self.panel["gp"].Refresh() # re-draw graph
        else:
            self.flagBlockUI = True
            if self.isRunning:
                self.timer["run"].Stop()
                self.isRunning = False
            self.vRW.getFrame(targetFI, self.callback, self.bp_sTxt)
    
    #-------------------------------------------------------------------
    
    def callback(self, rData, flag=""):
        """ call back function after running thread
        
        Args:
            rData (tuple): Received data from queue at the end of thread running
            flag (str): Indicator of origianl operation of this callback
        
        Returns:
            None
        """
        if DEBUG: print("ReviseCSV.callbackFunc()")
        
        if flag == "finalizeSavingVideo":
            msg = 'Saved.\n'
            msg += self.savVidFP 
            wx.MessageBox(msg, "Info", wx.OK|wx.ICON_INFORMATION)
            self.jumpToFrame(0) 
        # show current frame
        self.displayFrameImage(self.vRW.currFrame, flagMakeDispImg=True) 
        self.flagBlockUI = False
        self.panel["gp"].Refresh() # re-draw graph
    
    #-------------------------------------------------------------------
    
    def onSpace(self, event):
        """ start/stop continuous play 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("ReviseCSV.onSpace()")

        if self.flagBlockUI: return

        if self.isRunning == False:
            self.isRunning = True
            self.timer["run"] = wx.CallLater(5, self.play)
        else:
            try: # stop timer
                self.timer["run"].Stop() 
                self.timer["run"] = None
            except: pass
            self.isRunning = False
            
    #-------------------------------------------------------------------
    
    def play(self):
        """ load the next frame and move forward if it's playing.

        Args: None

        Returns: None
        """ 
        if DEBUG: print("ReviseCSV.play()")

        self.jumpToFrame(-1) # load next frame
        
        if self.isRunning:
            if self.vRW.fi >= self.endDataIdx: # reached end of available data
                self.onSpace(None) # stop 
            else:
                self.timer["run"] = wx.CallLater(5, self.play) # to next frame

    #-------------------------------------------------------------------
    
    def displayFrameImage(self, frameImg, flagMakeDispImg=True):
        """ Display a frame image in StaticBitmap object.
        
        Args:
            frameImg (numpy.ndarray): frame image
            flagMakeDispImg (bool): whether to process frame image

        Returns:
            None
        """ 
        if DEBUG: print("ReviseCSV.displayFrameImage()")

        img = self.makeDispImg(frameImg) # make image to display 
        
        ### display image 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        wxImg = wx.Image(img.shape[1], img.shape[0])
        wxImg.SetData(img.tostring())
        self.dispImg_sBmp.SetBitmap(wxImg.ConvertToBitmap())

    #-------------------------------------------------------------------
    
    def makeDispImg(self, frameImg):
        """ Make an image for UI or saving video.
        
        Args:
            frameImg (numpy.ndarray): frame image.

        Returns:
            img (numpy.ndarray): frame image with some drawing and resizing. 
        """ 
        if DEBUG: print("ReviseCSV.makeDispImg()")

        img = frameImg.copy()
        
        if self.ratFImgDispImg == None:
            self.ratFImgDispImg = calcI2DIRatio(self.vRW.currFrame, 
                                                self.pi['mp']['sz'])
        
        hD = self.oData[self.vRW.fi][self.hdi]
        hD = str2num(hD, 'int')
        if type(hD) == int:
            pts = []
            pts.append([self.oData[self.vRW.fi][self.hxi], 
                        self.oData[self.vRW.fi][self.hyi]]) 
            pts.append([self.oData[self.vRW.fi][self.bxi], 
                        self.oData[self.vRW.fi][self.byi]])
            for i in range(len(pts)):
                for j in range(len(pts[i])):
                    pts[i][j] = str2num(pts[i][j], 'int')
                pts[i] = tuple(pts[i])
            r = 1.0/self.ratFImgDispImg
            lw = int(2 * r)
            cr = int(3 * r)
            if type(pts[0][0]) == int and type(pts[1][0]) == int:
                # line from hPos to bPos
                cv2.line(img, pts[0], pts[1], (0,255,0), lw)
                # dot on hPos
                cv2.circle(img, pts[0], cr, (0,125,255), -1)
         
        ### resize image
        if self.ratFImgDispImg != 1.0:
            img = cv2.resize(img, 
                             (0,0), 
                             fx=self.ratFImgDispImg, 
                             fy=self.ratFImgDispImg)
        ### write status (frame-index, number-of-frames, etc) 
        status_msg = "%i/ %i, %s"%(self.vRW.fi, 
                                   self.vRW.nFrames-1, 
                                   self.oData[self.vRW.fi][self.hdi])
        cv2.putText(img, # image
                    status_msg, # string
                    (5, 20), # bottom-left
                    cv2.FONT_HERSHEY_PLAIN, # fontFace
                    self.fImgFontScale, # fontScale
                    self.fImgFontCol, # font color
                    2) # thickness
        return img
    
    #-------------------------------------------------------------------
    
    def save(self):
        """ Save revised CSV result as another CSV file

        Args: None

        Returns: None
        """
        if DEBUG: print("ReviseCSV.save()")

        fh = open(self.csvFP, 'r')
        lines = fh.readlines()
        fh.close()
        timestamp = get_time_stamp().replace("_","")[:14]
        # new file to write
        fp = self.csvFP.replace(".csv", "_rev_%s.csv"%(timestamp)) 
        fh = open(fp, 'w')
        flagDataStarted = False
        for line in lines:
            items = [ x.strip() for x in line.split(',') ]
            if flagDataStarted: # data started
                ### write data 
                for fi in range(self.vRW.nFrames):
                    newLine = ""
                    for ci in range(len(self.dataCols)):
                        newLine += "%s, "%(str(self.oData[fi][ci]))
                    newLine = newLine.rstrip(", ") + "\n"
                    fh.write(newLine) # write frame data
                break # finish writing data
            else: # write (parameter,column-title,...) lines before data lines
                if line.startswith("Timestamp"):
                    # write new timestamp
                    fh.write("Timestamp, %s\n"%(get_time_stamp()))
                else:
                    fh.write(line)
            if line.startswith("frame-index"): flagDataStarted = True
        _txt = "-----\n"
        fh.write(_txt)
        fh.close()

        msg = 'Saved.\n'
        msg += fp
        wx.MessageBox(msg, "Info", wx.OK|wx.ICON_INFORMATION)

    #-------------------------------------------------------------------
    
    def saveVideo(self):
        """ Save video (with revised head direction line)
        
        Args: None
        
        Returns: None
        """
        if DEBUG: print("ReviseCSV.saveVideo()")

        w = int(np.ceil(self.vRW.currFrame.shape[1]*self.ratFImgDispImg))
        h = int(np.ceil(self.vRW.currFrame.shape[0]*self.ratFImgDispImg))
        video_fSz = (w, h) # output video frame size
        timestamp = get_time_stamp().replace("_","")[:14]
        if self.vRW.vRecVideoCodec in ['avc1', 'h264']: ext = ".mp4"
        elif self.vRW.vRecVideoCodec == 'xvid': ext = ".avi"
        self.savVidFP = self.csvFP.replace(".csv", "_rev_%s%s"%(timestamp, ext))
        self.vRW.initWriter(self.savVidFP, 
                            video_fSz, 
                            self.callback, 
                            self.makeDispImg,
                            self.bp_sTxt)
        self.flagBlockUI = True 
    
    #-------------------------------------------------------------------
    
    def playSnd(self, flag=""):
        """ Play sound 

        Args:
            flag (str): Which sound to play.

        Returns:
            None
        """ 
        if DEBUG: print("ReviseCSV.playSnd()")

        if flag == "leftClick":
            ### play click sound
            snd_click = wx.adv.Sound("snd_click.wav")
            snd_click.Play(wx.adv.SOUND_ASYNC)

    #-------------------------------------------------------------------

#=======================================================================

class ReviseCSVApp(wx.App):
    def OnInit(self):
        if DEBUG: print("ReviseCSVApp.OnInit()")
        self.frame = ReviseCSVFrame()
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
        app = ReviseCSVApp(redirect = False)
        app.MainLoop()

