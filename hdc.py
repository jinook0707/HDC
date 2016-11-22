# coding: UTF-8

'''
Head-direction Coder v.0.1
jinook.oh@univie.ac.at
Cognitive Biology Dept., University of Vienna
- 2016.08

----------------------------------------------------------------------
Copyright (C) 2015 Jinook Oh, W. Tecumseh Fitch 
- Contact: jinook.oh@univie.ac.at, tecumseh.fitch@univie.ac.at

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
----------------------------------------------------------------------

1) Left mouse click: Set the head rect by click-and-drag 
            
2) Right mouse click: Set the head rect as -1 (when it's not available)

3) Left arrow key: Move to previous frame (-1)
4) Shift + Left: Move to a frame (-10)
5) Cmd + Left: Move to a frame (-100)

6) Right arrow key: Move to next frame (+1)
7) Shift + Right: Move to a frame (+10)
8) Cmd + Right: Move to a frame (+100)

9) Spacebar: Continue to proceed to next frames

10) M: Toggle 'manual input' (Script will keep previously determined head direction until this is off)

11) Cmd + S: Save CSV result data of head direction

12) Cmd + Q: Quit the app
'''

import Queue, plistlib
from threading import Thread
from os import getcwd, path
from sys import argv
from copy import copy
from time import time, sleep
from datetime import timedelta
from glob import glob
from random import shuffle

import wx
import cv2
import numpy as np

from modules.misc_funcs import GNU_notice, get_time_stamp, writeFile, show_msg, load_img, cvImg_to_wxBMP, calc_angle_diff  
from modules.cv_proc import CVProc

# ======================================================

class HDCFrame(wx.Frame):
    def __init__(self):
        self.w_size = (1200, 750)
        wx.Frame.__init__(self, None, -1, 'Head-direction Coder', size=self.w_size) # init frame
        self.SetPosition( (0, 25) )
        self.Show(True)
        self.panel = wx.Panel(self, pos=(0,0), size=self.w_size)
        self.panel.SetBackgroundColour('#000000')
        self.cv_proc = CVProc(self) 

        ### init variables
        self.msg_q = Queue.Queue()
        self.program_start_time = time()
        self.session_start_time = -1
        self.oData = {} # output data
        self.fPath = '' # folder path including frame images
        self.fi = 0 # current frame index
        self.frame_cnt = 0
        self.vFPS = 60 # fps for video file
        self.is_running = False # analysis is running by pressing spacebar
        self.timer_run = None # timer for running analysis
        self.is_lb_pressed = False # whether left mouse button is pressed or not
        self.animalSp = '' # animal species
        self.imgType = '' # display image type
        self.filenames = 'f%06i.jpg'

        ### user interface setup
        posX = 5
        posY = 10
        btn_width = 150
        b_space = 30
        self.btn_start = wx.Button(self.panel, -1, label='Analyze video (folder)', pos=(posX,posY), size=(btn_width, -1))
        self.btn_start.Bind(wx.EVT_LEFT_UP, self.onStartStopAnalyzeVideo)
        posY += b_space
        self.btn_quit = wx.Button(self.panel, -1, label='QUIT', pos=(posX,posY), size=(btn_width, -1))
        self.btn_quit.Bind(wx.EVT_LEFT_UP, self.onClose)
        
        ### Elapsed time
        posX = 170
        posY = 15
        self.sTxt_pr_time = wx.StaticText(self.panel, -1, label='0:00:00', pos=(posX, posY)) # elapsed time since program starts
        _x = self.sTxt_pr_time.GetPosition()[0] + self.sTxt_pr_time.GetSize()[0] + 15
        _stxt = wx.StaticText(self.panel, -1, label='since program started', pos=(_x, posY))
        _stxt.SetForegroundColour('#CCCCCC')
        self.font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.NORMAL)
        self.sTxt_pr_time.SetFont(self.font)
        self.sTxt_pr_time.SetBackgroundColour('#000000')
        self.sTxt_pr_time.SetForegroundColour('#00FF00')
        posY += b_space
        self.sTxt_s_time = wx.StaticText(self.panel, -1, label='0:00:00', pos=(posX, posY)) # elapsed time since session starts
        _x = self.sTxt_s_time.GetPosition()[0] + self.sTxt_s_time.GetSize()[0] + 15
        _stxt = wx.StaticText(self.panel, -1, label='since session started', pos=(_x, posY))
        _stxt.SetForegroundColour('#CCCCCC')
        self.sTxt_s_time.SetFont(self.font)
        self.sTxt_s_time.SetBackgroundColour('#000000')
        self.sTxt_s_time.SetForegroundColour('#CCCCFF')
       
        posX = _stxt.GetPosition()[0] + _stxt.GetSize()[0] + 30
        _posX = copy(posX)
        posY = self.sTxt_pr_time.GetPosition()[1]
        _sTxt= wx.StaticText(self.panel, -1, label='Animal: ', pos=(posX, posY))
        _sTxt.SetForegroundColour('#CCCCCC')
        posX += _sTxt.GetSize()[0]
        self.cho_animalSp = wx.Choice(self.panel, id=-1, pos=(posX, posY-2), choices=['Marmoset', 'Alligator', 'Gerbil'])
        self.cho_animalSp.SetSelection(0)
        self.cho_animalSp.Bind(wx.EVT_CHOICE, self.onChangeAnimalSp)
        posX += self.cho_animalSp.GetSize()[0] + 15
        _sTxt= wx.StaticText(self.panel, -1, label='Display-image: ', pos=(posX, posY))
        _sTxt.SetForegroundColour('#CCCCCC')
        posX += _sTxt.GetSize()[0]
        self.cho_imgType = wx.Choice(self.panel, id=-1, pos=(posX, posY-2), choices=['RGB-image', 'Greyscale(Diff)', 'Greyscale(Edge)'])
        self.cho_imgType.SetSelection(0)
        self.cho_imgType.Bind(wx.EVT_CHOICE, self.onChangeImgType)
        posX += self.cho_imgType.GetSize()[0] + 30
        self.chk_manual = wx.CheckBox(self.panel, id=-1, pos=(posX, posY), label='')
        posX += self.chk_manual.GetSize()[0]
        _sTxt= wx.StaticText(self.panel, -1, label='Continuous manual input', pos=(posX, posY))
        _sTxt.SetForegroundColour('#CCCCCC')
        posX = _posX 
        posY += b_space 
        self.sTxt_fps = wx.StaticText(self.panel, -1, label='FPS', pos=(posX, posY)) # FPS
        self.sTxt_fps.SetForegroundColour('#CCCCCC')
        posX = self.sTxt_fps.GetPosition()[0] + self.sTxt_fps.GetSize()[0] + 50
        self.sTxt_fp = wx.StaticText(self.panel, -1, label='FOLDER NAME', pos=(posX, posY)) # folder path
        self.sTxt_fp.SetForegroundColour('#CCCCCC')
        posX = self.sTxt_fps.GetPosition()[0]

        ### Frame image
        self.loaded_img_pos = (5, self.sTxt_s_time.GetPosition()[1]+self.sTxt_s_time.GetSize()[1]+20)
        self.loaded_img_sz = (self.w_size[0]-10, self.w_size[1]-self.loaded_img_pos[1]-5)
        self.loaded_img = wx.StaticBitmap( self.panel, -1, wx.NullBitmap, self.loaded_img_pos, self.loaded_img_sz )
        self.loaded_img.Bind(wx.EVT_ERASE_BACKGROUND, lambda evt: None) 
        self.loaded_img.Bind(wx.EVT_LEFT_DOWN, self.onMouseLeftDown)
        self.loaded_img.Bind(wx.EVT_LEFT_UP, self.onMouseLeftUp)
        self.loaded_img.Bind(wx.EVT_MOTION, self.onMouseMove)
        self.loaded_img.Bind(wx.EVT_RIGHT_UP, self.onMouseRightUp)
        
        statbar = wx.StatusBar(self, -1)
        self.SetStatusBar(statbar)

        ### keyboard binding
        exit_btnId = wx.NewId()
        save_btnId = wx.NewId()
        left_btnId = wx.NewId(); right_btnId = wx.NewId()
        leftJump_btnId = wx.NewId(); rightJump_btnId = wx.NewId()
        leftJumpFurther_btnId = wx.NewId(); rightJumpFurther_btnId = wx.NewId()
        resizeRectUp_btnId = wx.NewId(); resizeRectDown_btnId = wx.NewId(); resizeRectLeft_btnId = wx.NewId(); resizeRectRight_btnId = wx.NewId()
        space_btnId = wx.NewId() 
        mi_btnId = wx.NewId()
        self.Bind(wx.EVT_MENU, self.onClose, id = exit_btnId)
        self.Bind(wx.EVT_MENU, self.onSave, id = save_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onLeft(event, 'left'), id=left_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onLeft(event, 'leftjump'), id=leftJump_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onLeft(event, 'leftjumpfurther'), id=leftJumpFurther_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onRight(event, 'right'), id=right_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onRight(event, 'rightjump'), id=rightJump_btnId)
        self.Bind(wx.EVT_MENU, lambda event: self.onRight(event, 'rightjumpfurther'), id=rightJumpFurther_btnId)
        self.Bind(wx.EVT_MENU, self.onSpace, id = space_btnId)
        self.Bind(wx.EVT_MENU, self.onCheckManualInput, id = mi_btnId)
        accel_tbl = wx.AcceleratorTable([ (wx.ACCEL_CMD,  ord('Q'), exit_btnId ), 
                                          (wx.ACCEL_CMD,  ord('S'), save_btnId ),
                                          (wx.ACCEL_NORMAL,  wx.WXK_RIGHT, right_btnId ), 
                                          (wx.ACCEL_NORMAL,  wx.WXK_LEFT, left_btnId ), 
                                          (wx.ACCEL_SHIFT,  wx.WXK_RIGHT, rightJump_btnId ), 
                                          (wx.ACCEL_SHIFT,  wx.WXK_LEFT, leftJump_btnId ),
                                          (wx.ACCEL_NORMAL, wx.WXK_SPACE, space_btnId),
                                          (wx.ACCEL_NORMAL,  ord('M'), mi_btnId), 
                                          (wx.ACCEL_CMD, wx.WXK_LEFT, leftJumpFurther_btnId), 
                                          (wx.ACCEL_CMD, wx.WXK_RIGHT, rightJumpFurther_btnId),
                                           ]) 
        self.SetAcceleratorTable(accel_tbl)
        
        ### set timer for updating the current running time
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimer, self.timer)
        self.timer.Start(1000)

        self.Bind( wx.EVT_CLOSE, self.onClose )

    # --------------------------------------------------       
      
    def onLeft(self, event, flag):
        ''' left arrow key is pressed. go backward in the series of images
        '''
        if self.fPath == '' or self.fi == 1: return
        if self.is_running == True: # if continuous analysis is running
            if event != None: # if user pressed this manual navigation key
                self.onSpace(None) # stop continous analysis
        if flag == 'left': self.fi -= 1
        elif flag == 'leftjump': self.fi = max(1, self.fi-10)
        elif flag == 'leftjumpfurther': self.fi = max(1, self.fi-100)
        self.proc_img()

    # --------------------------------------------------       
    
    def onRight(self, event, flag):
        ''' right arrow key is pressed. go forward in the series of images
        '''
        if self.fPath == '' or self.fi >= self.frame_cnt: return
        if self.is_running == True: # if continuous analysis is running 
            ### FPS update
            self.fps += 1
            if time()-self.last_fps_time >= 1:
                self.sTxt_fps.SetLabel( "FPS: %i"%(self.fps) )
                self.fps = 0
                self.last_fps_time = time()
            if event != None: # user pressed this manual navigation key
                self.onSpace(None) # stop continuous analysis
        if flag == 'right': self.fi += 1
        elif flag == 'rightjump': self.fi = min(self.frame_cnt, self.fi+10)
        elif flag == 'rightjumpfurther': self.fi = min(self.frame_cnt, self.fi+100) 
        if self.fi == self.frame_cnt and self.is_running == True: self.onSpace(None)
        self.proc_img()

    #------------------------------------------------
    
    def onSpace(self, event):
        ''' start/stop continuous frame analysis
        '''
        if self.fPath == '' or self.fi > self.frame_cnt: return
        if self.is_running == False:
            self.is_running = True
            self.fps = 0
            self.last_fps_time = time()
            self.timer_run = wx.FutureCall(1, self.onRight, None, 'right')
        else:
            try: # stop timer
                self.timer_run.Stop() 
                self.timer_run = None
            except: pass
            self.sTxt_fps.SetLabel('')
            self.is_running = False # stop continuous analysis
            
    #------------------------------------------------
    
    def onMouseLeftDown(self, event):
        if self.fPath == '': return
        mp = event.GetPosition()
        self.bPos = (mp[0], mp[1]) # base position (to determine head direction and position)
        self.is_lb_pressed = True
        fp = path.join(self.fPath, self.filenames%self.fi)
        self.tmp_img = load_img(fp, flag='cv')
        self.tmp_img = self.onLoadImg(self.tmp_img)
    
    #------------------------------------------------

    def onMouseMove(self, event):
        if self.fPath == '': return
        if self.is_lb_pressed == True:
            mp = event.GetPosition()
            hPos = (mp[0], mp[1])
            img_ = self.tmp_img.copy()
            rImg, __, __, __ = self.cv_proc.proc_img( img_, img_, hPos, self.bPos )
            self.loaded_img.SetBitmap( cvImg_to_wxBMP(rImg) ) # display image
    
    #------------------------------------------------

    def onMouseLeftUp(self, event):
        if self.fPath == '': return
        mp = event.GetPosition()
        hPos = (mp[0], mp[1])
        self.oData[self.fi]['mHPos'] = True
        self.oData[self.fi]['mHD'] = True
        self.proc_img(hPos, self.bPos)
        self.is_lb_pressed = False
        self.bPos = None
        self.tmp_img = None

    #------------------------------------------------
 
    def onMouseRightUp(self, event):
        if self.fPath == '': return
        self.oData[self.fi]['hPos'] = ('D','D') # delete info
        self.oData[self.fi]['mHPos'] = True
        self.oData[self.fi]['bPos'] = ('D','D') # delete info
        self.oData[self.fi]['hD'] = 'D' # delete info
        self.oData[self.fi]['mHD'] = True
        self.proc_img()

    #------------------------------------------------
    
    def onChangeImgType(self, event):
        self.imgType = self.cho_imgType.GetString( self.cho_imgType.GetSelection() )
        self.proc_img()

    #------------------------------------------------
    
    def onChangeAnimalSp(self, event):
        self.animalSp = self.cho_animalSp.GetString( self.cho_animalSp.GetSelection() )
        if self.animalSp == 'Marmoset':
            self.cv_proc.mExOIter = 8 # number of iterations of morphologyEx (for reducing noise & minor features after absdiff from background image)
            self.cv_proc.thParam = 50 # Threshold parameter
            self.cv_proc.contourTh = 50 # minimum contour size
            self.cv_proc.motionTh = [100, 300] # lower and upper threshold for recognizing a motion in a frame. this number is a square root of sum(different_pixel_values)/255 
            self.cv_proc.degTh = 20 # if head direction differece is over this threshold, reject the calculated head direction and copy the previous frame's head direction 
            self.cv_proc.hdLineLen = 50 # length of head-direction line
        elif self.animalSp == 'Alligator':
            self.cv_proc.mExOIter = 0
            self.cv_proc.thParam = 50
            self.cv_proc.contourTh = 5
            self.cv_proc.motionTh = [80, 300]
            self.cv_proc.degTh = 30
            self.cv_proc.hdLineLen = 50 
            self.cv_proc.nKMC = 10 # number of clusters for k-means clustering
        elif self.animalSp == 'Gerbil': 
            self.cv_proc.mExOIter = 2
            self.cv_proc.thParam = 50
            self.cv_proc.contourTh = 5
            self.cv_proc.motionTh = [80, 300]
            self.cv_proc.degTh = 30
            self.cv_proc.hdLineLen = 30
            self.cv_proc.nKMC = 4
        if event != None: self.proc_img()

    #------------------------------------------------
    
    def onCheckManualInput(self, event):
        pass
    
    #------------------------------------------------
    
    def onLoadImg(self, img):
        ### resize if image is too large
        if img.shape[1] > self.loaded_img_sz[0]:
            _r = float(self.loaded_img_sz[0]) / img.shape[1]
            img = cv2.resize(img, (0,0), fx=_r, fy=_r)
        if img.shape[0] > self.loaded_img_sz[1]:
            _r = float(self.loaded_img_sz[1]) / img.shape[0]
            img = cv2.resize(img, (0,0), fx=_r, fy=_r)
        return img
    
    #------------------------------------------------
    
    def proc_img(self, hPos=None, bPos=None):
        if self.fPath == '': return
        fp = path.join(self.fPath, self.filenames%self.fi)
        img2 = load_img(fp, flag='cv')
        img2 = self.onLoadImg(img2)
        if self.fi > 1: 
            fp = path.join(self.fPath, self.filenames%(self.fi-1))
            img1 = load_img(fp, flag='cv')
        else:
            img1 = img2.copy()
        img1 = self.onLoadImg(img1)
        rIMG, rHPos, rBPos, rHD = self.cv_proc.proc_img(img1, img2, hPos, bPos, self.imgType, self.animalSp) # cv_proc.proc_img returns image, head position, head direction
        
        self.loaded_img.SetBitmap( cvImg_to_wxBMP(rIMG) ) # display image
        self.oData[self.fi]['hPos'] = rHPos # update the head position 
        self.oData[self.fi]['bPos'] = rBPos # update the base position 
        self.oData[self.fi]['hD'] = rHD # update the head direction
        if self.chk_manual.GetValue() == True: self.oData[self.fi]['mHD'] = True
        
        if self.is_running == True:
            #if 'None' not in [rHP[0], rHP[1]]: # all head information were returned properly
            if self.fi < self.frame_cnt: # there's more frames to run
                self.timer_run = wx.FutureCall(1, self.onRight, None, 'right') # keep continuous analysis

    #------------------------------------------------
    
    def onStartStopAnalyzeVideo(self, event):
        '''Choose a video file and starts analysis'''
        if self.session_start_time == -1: # not in analysis session. start a session
            dlg = wx.DirDialog(self, "Choose directory for analysis", getcwd(), wx.DD_DEFAULT_STYLE|wx.DD_DIR_MUST_EXIST)
            if dlg.ShowModal() == wx.ID_CANCEL: return
            self.fPath = dlg.GetPath()
            self.frame_cnt = len(glob(path.join(self.fPath, 'f*.jpg')))
            if self.frame_cnt == 0:
                show_msg('No jpg frame images in the chosen directory.')
                self.fPath = ''
                return
            if path.isfile( path.join(self.fPath, 'bg.jpg') ) == False:
                show_msg(msg="'bg.jpg' file is not found in the chosen directory.", cancel_btn=False)
                return
            self.cv_proc.bg = load_img( path.join(self.fPath, 'bg.jpg'), flag='cv' ) # load background image
            self.cv_proc.bg = self.onLoadImg(self.cv_proc.bg)
            self.onChangeAnimalSp(None) # set default parameters of cv_proc
            self.sTxt_fp.SetLabel( '%s'%(path.basename(self.fPath)) )
            result_csv_file = self.fPath + '.csv'
            if path.isfile(result_csv_file) == False: # result CSV file doesn't exist
                for i in range(1, self.frame_cnt+1):
                    self.oData[i] = dict( hPos=('None','None'), mHPos=False, bPos=('None','None'), hD='None', mHD=False ) # head position, head position was manually determined, base of head(around nect), head direction, head direction was manually determined
            else: # result CSV file exists
                f = open(result_csv_file, 'r')
                lines = f.readlines()
                f.close()
                for i in range(1, self.frame_cnt+1):
                    self.oData[i] = dict( hPos = ('None','None'), mHPos=False, bPos=('None','None'), hD = 'None', mHD=False )
                    if i < len(lines):
                        items = [ x.strip() for x in lines[i].split(',') ]
                        idx_ = int(items[0])
                        if items[1] == 'None': hPos_val = ('None', 'None')
                        elif items[1] == 'D': hPos_val = ('D', 'D')
                        else: hPos_val = ( int(items[1]), int(items[2]) )
                        if items[3] == 'True': mHPos_val = True
                        else: mHPos_val = False
                        if items[4] == 'None': bPos_val = ('None', 'None')
                        elif items[4] == 'D': bPos_val = ('D', 'D')
                        else: bPos_val = ( int(items[4]), int(items[5]) )
                        if items[6] == 'None': hD_val = 'None'
                        elif items[6] == 'D': hD_val = 'D'
                        else: hD_val = int(items[6])
                        if items[7] == 'True': mHD_val = True
                        else: mHD_val = False
                        self.oData[idx_]['hPos'] = copy(hPos_val)
                        self.oData[idx_]['mHPos'] = copy(mHPos_val)
                        self.oData[idx_]['bPos'] = copy(bPos_val)
                        self.oData[idx_]['hD'] = copy(hD_val)
                        self.oData[idx_]['mHD'] = copy(mHD_val)
            self.fi = 1
            self.session_start_time = time()
            self.btn_start.SetLabel('Stop analysis')
            ### start video recorder
            self.video_path = self.fPath + '.avi'
            fp = path.join(self.fPath, self.filenames%self.fi)
            img = load_img(fp, flag='cv')
            img = self.onLoadImg(img)
            self.cv_proc.start_video_rec( self.video_path, img )
            self.proc_img() # process 1st image
        else: # in session. stop it.
            result = show_msg(msg='Save data?', cancel_btn = True)
            if result == True: self.onSave(None)
            if self.is_running == True: self.onSpace(None)
            self.session_start_time = -1
            self.sTxt_s_time.SetLabel('0:00:00')
            self.btn_start.SetLabel('Analyze video (folder)')
            self.sTxt_fp.SetLabel('')
            self.loaded_img.SetBitmap(wx.NullBitmap)
            self.fPath = ''
            self.frame_cnt = 0
            self.fi = 0
            self.oData = {}
            self.cv_proc.stop_video_rec()
    
    # --------------------------------------------------       
    
    def onSave(self, event):
        nfHPM = 0 # number of frames in which head position is missing
        nfHDM = 0 # number of frames in which head direction is missing
        fp_ = self.fPath + '.csv'
        fh = open(fp_, 'w')
        fh.write('frame-index, hPosX, hPosY, mHPos, bPosX, bPosY, hDir, mHD\n')
        for fi in range(1, self.frame_cnt+1):
            d_ = self.oData[fi]
            if d_['hPos'] == ('None','None') or d_['hPos'] == ('D','D'): nfHPM += 1 # head position is missing
            if type(d_['hD']) != int: nfHDM += 1 # head direction is missing
            line = '%i, %s, %s, %s, %s, %s, %s, %s\n'%(fi,
                                                       str(d_['hPos'][0]), 
                                                       str(d_['hPos'][1]), 
                                                       str(d_['mHPos']),
                                                       str(d_['bPos'][0]), 
                                                       str(d_['bPos'][1]), 
                                                       str(d_['hD']), 
                                                       str(d_['mHD'])
                                                       )
            fh.write(line)
        fh.write('------------------------------------------------------------------\n')
        fh.write('Number of frames in which head position is missing, %i\n'%nfHPM)
        fh.write('Number of frames in which head direction is missing, %i\n'%nfHDM)
        fh.close()

        msg = 'Saved.\n'
        chr_num = 50 # characters in one line
        if len(fp_) > chr_num:
            for i in range(len(fp_)/chr_num):
                msg += '%s\n'%(fp_[chr_num*i:chr_num*(i+1)])
            msg += '%s\n'%(fp_[chr_num*(i+1):])
        show_msg(msg, size=(400,200))

    # --------------------------------------------------       

    def onTimer(self, event):
        ''' Main timer 
        updating running time on the main window
        '''
        ### update several running time
        e_time = time() - self.program_start_time
        self.sTxt_pr_time.SetLabel( str(timedelta(seconds=e_time)).split('.')[0] )
        if self.session_start_time != -1:
            e_time = time() - self.session_start_time
            self.sTxt_s_time.SetLabel( str(timedelta(seconds=e_time)).split('.')[0] )

    # --------------------------------------------------

    def show_msg_in_statbar(self, msg, time=5000):
        self.SetStatusText(msg)
        wx.FutureCall(time, self.SetStatusText, "") # delete it after a while

    # --------------------------------------------------

    def onClose(self, event):
        self.timer.Stop()
        result = True
        if self.session_start_time != -1: # session is running
            result = show_msg(msg='Session is not stopped..\nUnsaved data will be lost. (Stop analysis or Cmd+S to save.)\nOkay to proceed to exit?', cancel_btn = True)
        if result == True:
            if self.cv_proc.video_rec != None: self.cv_proc.stop_video_rec()
            wx.FutureCall(500, self.Destroy)

# ======================================================

class HDCApp(wx.App):
    def OnInit(self):
        self.frame = HDCFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

# ======================================================

if __name__ == '__main__':
    if len(argv) > 1:
        if argv[1] == '-w': GNU_notice(1)
        elif argv[1] == '-c': GNU_notice(2)
    else:
        GNU_notice(0)
        CWD = getcwd()
        app = HDCApp(redirect = False)
        app.MainLoop()




