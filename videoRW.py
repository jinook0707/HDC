# coding: UTF-8

import queue
from threading import Thread 
from os import path, remove

import numpy as np
import cv2, wx

from fFuncNClasses import receiveDataFromQueue

DEBUG = False 

#=======================================================================

class VideoRW:
    """ Class for reading/writing frame from/to video file 

    Args:
        parent: parent object

    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, parent):
        if DEBUG: print("VideoRW.__init__()")
        
        ##### [begin] setting up attributes -----
        self.p = parent
        self.fPath = "" # file path of video
        self.vCap = None # VideoCapture object of OpenCV
        self.vCapFSz = (-1, -1) # frame size (w, h) of frame image 
          # of current video
        self.currFrame = None # current frame image (ndarray)
        self.nFrames = 0 # total number of frames
        self.fi = -1 # current frame index
        self.th = None # thread
        self.q2m = queue.Queue() # queue from thread to main
        self.q2t = queue.Queue() # queue from main to thread 
        self.timer = {} # timers for this class
        self.vRecVideoCodec = "avc1" # video codec for saving analysis screen 
          # (h264/avc1 (.mp4) or xvid (.avi))
        self.vRecFPS = 60 # fps for analysis video file
        ##### [end] setting up attributes -----

    #-------------------------------------------------------------------
    
    def initReader(self, fPath):
        """ init. video file reading 

        Args:
            fPath (str): Path of video file to read. 

        Returns:
            None
        """
        if DEBUG: print("VideoRW.initReader()") 

        self.fPath = fPath
        # init video capture
        self.vCap = cv2.VideoCapture(fPath)
        # get total number of frames
        self.nFrames = int(self.vCap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fi = -1 
        self.getFrame(-1) # read the 1st frame
        # store frame size
        self.vCapFSz = (self.currFrame.shape[1], self.currFrame.shape[0]) 

    #-------------------------------------------------------------------
    
    def initWriter(self, fPath, video_fSz, 
                   callbackFunc=None, procFunc=None, sTxt=None):
        """ init. video file writing 

        Args:
            fPath (str): Path of video file to record.
            video_fSz (tuple): Video frame size to record.
            callbackFunc (function): Callback function to call after writing.
            procFunc (function): Function to call to process 
              before saving each frame image.
            sTxt (wx.StaticText): StaticText to show navigation progress,
              when navigating with thread.

        Returns:
            None
        """
        if DEBUG: print("VideoRW.initWriter()") 

        # if file already exists, delete it
        if path.isfile(fPath): remove(fPath)
        
        self.writingVideoFrameSz = video_fSz 
        self.callbackFunc = callbackFunc
        fourcc = cv2.VideoWriter_fourcc(*'%s'%(self.vRecVideoCodec))
        # init video writer
        self.video_rec = cv2.VideoWriter(fPath, 
                                         fourcc=fourcc, 
                                         fps=self.vRecFPS, 
                                         frameSize=video_fSz, 
                                         isColor=True)
        if callbackFunc != None:
            ### set timer for updating current frame index 
            self.sTxt = sTxt # wx.StaticText to show progress
            self.timer["writeFrames"] = wx.Timer(self.p)
            self.p.Bind(wx.EVT_TIMER,
                        lambda event: self.onTimer(event, "writeFrames"),
                        self.timer["writeFrames"])
            self.timer["writeFrames"].Start(10) 
            ### start thread to write 
            self.callbackFunc = callbackFunc # store callback function
            self.th = Thread(target=self.writeFrames, 
                             args=(self.video_rec, self.q2m, procFunc,))
            wx.CallLater(20, self.th.start)
                
    #-------------------------------------------------------------------
    
    def getFrame(self, targetFI=-1, callbackFunc=None, sTxt=None):
        """ Retrieve a frame image with a given index or 
        just the next frame when index is not given. 

        Args:
            targetFI (int): Target frame index to retrieve.
            callbackFunc (None/function): Callback function when targetFI
              is not -1, meaning it'd be thread running.
            sTxt (wx.StaticText): StaticText to show navigation progress,
              when navigating with thread.

        Returns:
            ret (bool): Notifying whether the image retrieval was successful.
            frame (numpy.ndarray): Frame image.
        """
        if DEBUG: print("VideoRW.getFrame()")

        if self.fi >= self.nFrames: return
        
        if targetFI == -1:
        # target index is not given
            for i in range(self.fi, self.nFrames):
                ret, frame = self.vCap.read() # read next frame
                self.fi += 1
                if ret: break # stop reading, if it was successful
            if ret:
                self.currFrame = frame
            else: # failed to retrieve frame
                frame = np.zeros(self.vCapFSz) # return blank image
        else:
        # target index is given
            if targetFI > self.fi:
                nRead = targetFI - self.fi 
            elif targetFI < self.fi:
                self.vCap.release()
                self.vCap = cv2.VideoCapture(self.fPath)
                self.fi = -1 
                nRead = targetFI + 1
            else:
                return
            ### start thread to read
            self.th = Thread(target=self.readFrames, 
                             args=(self.fi, nRead, self.q2m,))
            self.th.start()
            self.callbackFunc = callbackFunc # store callback function
            self.sTxt = sTxt # wx.StaticText to show progress
            self.targetFI = targetFI
            ### set timer for updating current frame index 
            self.timer["readFrames"] = wx.Timer(self.p)
            self.p.Bind(wx.EVT_TIMER,
                        lambda event: self.onTimer(event, "readFrames"),
                        self.timer["readFrames"])
            self.timer["readFrames"].Start(10) 
    
    #-------------------------------------------------------------------

    def readFrames(self, fi, n, q2m):
        """ read frames from video
        
        Args:
            fi (int): Current frame index
            n (int): Number of frames to read
            q2m (queue.Queue): Queue to send data to main
        
        Returns:
            None
        """
        if DEBUG: print("VideoRW.readFrames()") 
        
        for i in range(n):
            ret, f = self.vCap.read()
            if not ret: break
            frame = f
            q2m.put((fi,), True, None)
            fi += 1
        q2m.put((fi, frame), True, None)

    #-------------------------------------------------------------------
    
    def writeFrames(self, video_rec, q2m, procFunc):
        """ Write frames to VideoRecorder to save.

        Args:
            video_rec (cv2.VideoWriter)
            q2m (queue.Queue): Queue to send data to main thread.
            procFunc (function): Function to call before saving.

        Returns:
            None
        """
        if DEBUG: print("VideoRW.writeFrames()")
        
        ### write video frames
        for fi in range(self.nFrames):
            if fi > 0: self.getFrame(-1)
            frame = self.currFrame.copy()
            if procFunc != None: frame = procFunc(frame)
            video_rec.write(frame) # write a frame
            msg = "Writing video.. frame-idx: %i/%i"%(fi, self.nFrames-1)
            q2m.put((msg,), True, None)
        q2m.put((msg, frame), True, None)
    
    #-------------------------------------------------------------------
    
    def writeFrame(self, frame):
        """ Write a single frame to VideoRecorder to save.

        Args: None

        Returns: None
        """
        if DEBUG: print("VideoRW.writeFrame()")

        self.video_rec.write(
                            cv2.resize(frame, self.writingVideoFrameSz)
                            ) # write a frame
    
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
        
        if flag == "readFrames":
        # navigating (reading frames) to a specific frame
            if len(rData) == 1:
                self.sTxt.SetLabel("Frame-index: %i"%(rData[0]))
            elif len(rData) == 2:
            # reached target frame index
                self.fi, self.currFrame = rData
                self.timer["readFrames"].Stop()
                self.timer["readFrames"] = None
                self.targetFI = -1
                self.sTxt.SetLabel("-")
                self.th.join()
                self.th = None
                self.callbackFunc(rData)
        
        elif flag == "writeFrames":
            if len(rData) == 1:
                if self.sTxt != None: self.sTxt.SetLabel(rData[0])
            elif len(rData) == 2:
                self.timer["writeFrames"].Stop()
                self.timer["writeFrames"] = None
                if self.sTxt != None: self.sTxt.SetLabel("")
                self.closeWriter() 
                self.callbackFunc(rData, "finalizeSavingVideo") 
    
    #-------------------------------------------------------------------

    def closeReader(self):
        """ close videoCapture 

        Args: None

        Returns: None
        """
        if DEBUG: print("VideoRW.closeReader()") 
        
        self.vCap.release() # close video capture instance
        self.vCap = None
        self.fPath = ""
        self.fi = -1
        self.nFrames = 0

    #-------------------------------------------------------------------

    def closeWriter(self):
        """ close videoWriter

        Args: None

        Returns: None
        """
        if DEBUG: print("VideoRW.closeWriter()")

        # finish recorder
        self.video_rec.release()
        self.video_rec = None
        
    #-------------------------------------------------------------------
#=======================================================================

if __name__ == "__main__": pass
