'''
----------------------------------------------------------------------
Copyright (C) 2016 Jinook Oh, W. Tecumseh Fitch 
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
'''

from time import time, sleep
from copy import copy
from os import path
from glob import glob
from random import randint

import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans 
from scipy.cluster.hierarchy import fclusterdata

from modules.misc_funcs import get_time_stamp, writeFile, chk_msg_q, calc_pt_line_dist, calc_line_angle, calc_angle_diff, load_img

flag_window = True # create an opencv window or not
flag_video_rec = False # video recording

# ======================================================

class CVProc:
    def __init__(self, parent):
        self.parent = parent

        self.fourcc = cv2.cv.CV_FOURCC('x', 'v', 'i', 'd')
        #self.fourcc = cv2.VideoWriter_fourcc('x', 'v', 'i', 'd') #cv2.cv.CV_FOURCC('x', 'v', 'i', 'd')
        self.video_rec = None # video recorder
        self.fSize = (960, 540) # default frame size
        self.gmpts = [] # good matched points found by ORB
        self.cluster_cols = [(125,125,125), (0,0,255), (255,0,0), (0,255,255), (255,100,100), 
                             (0,255,0), (0,0,0), (50,255,50), (100,100,255), (0,125,255)]
        self.video_rec = None

    # --------------------------------------------------

    def start_video_rec(self, video_fp, frame_arr):
        self.fSize= (frame_arr.shape[1], frame_arr.shape[0])
        self.video_fSize = (int(self.fSize[0]/2), int(self.fSize[1]/2)) # output video frame size
        self.video_rec = cv2.VideoWriter( video_fp, self.fourcc, self.parent.vFPS, self.video_fSize, 1 )

    # --------------------------------------------------

    def stop_video_rec(self):
        self.video_rec.release()
        self.video_rec = None

    # --------------------------------------------------
    
    def proc_img(self, prev_frame_arr, frame_arr, hPos=None, bPos=None, imgType='RGB-image', animalSp='Marmoset'):
        hD = None
        status_msg = "%i/ %i, "%(self.parent.fi, self.parent.frame_cnt)
        
        ### find the subject's body (or its parts) by comparing the current frame with the background image 
        diff = cv2.absdiff( frame_arr, self.bg ) # difference from the background image 
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) 
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=self.mExOIter) # decrease noise & minor features
        #diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=self.mExCIter) # closing small holes (useful for intensify minor features)
        __, diff = cv2.threshold(diff, self.thParam, 255, cv2.THRESH_BINARY) # make the recognized parts clear

        ### edge detection on the recognized parts
        edged = cv2.Canny(diff, 150, 150)
        (cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # get contours of difference
        cnt_info = [] # contour info (size, center-X, center-Y)
        cntsBr = [] # bounding rect(x1,y1,x2,y2) for all contours
        for ci in range(len(cnts)):
            mr = cv2.boundingRect(cnts[ci])
            if mr[2]+mr[3] < self.contourTh: continue
            cv2.circle(edged, (mr[0]+mr[2]/2,mr[1]+mr[3]/2), mr[2]/2, 125, 1)
            cnt_info.append( (mr[2]+mr[3], mr[0]+mr[2]/2, mr[1]+mr[3]/2) )
            if cntsBr == []: cntsBr = [mr[0], mr[1], mr[0]+mr[2], mr[1]+mr[3]]
            else:
                if mr[0] < cntsBr[0]: cntsBr[0] = mr[0]
                if mr[1] < cntsBr[1]: cntsBr[1] = mr[1]
                if mr[0]+mr[2] > cntsBr[2]: cntsBr[2] = mr[0]+mr[2]
                if mr[1]+mr[3] > cntsBr[3]: cntsBr[3] = mr[1]+mr[3]
        if cntsBr == []: cpt = (-1, -1)
        else: cpt = ( cntsBr[0]+(cntsBr[2]-cntsBr[0])/2, cntsBr[1]+(cntsBr[3]-cntsBr[1])/2 ) # center point of all contours
        if self.parent.fi == 1: self.last_motion_frame = frame_arr.copy()
        
        if self.parent.fi > 1:
            self.prev_hd = self.parent.oData[self.parent.fi-1]["hD"] # head direction of previous frame
            self.prev_hPos = self.parent.oData[self.parent.fi-1]["hPos"] # head position of previous frame
            self.prev_bPos = self.parent.oData[self.parent.fi-1]["bPos"] # base position of previous frame
        else:
            self.prev_hd = None
            self.prev_hPos = None
            self.prev_bPos = None
        
        if hPos != None and bPos != None: # head postion and base position is given by user
            hD = calc_line_angle(bPos, hPos)
        elif self.parent.oData[self.parent.fi]["mHD"] == True: # head direction is manually fixed
            hD = self.parent.oData[self.parent.fi]["hD"]
            hPos = self.parent.oData[self.parent.fi]["hPos"]
            bPos = self.parent.oData[self.parent.fi]["bPos"]
        elif self.parent.chk_manual.GetValue() == True: # manual input is checked
            hD = self.prev_hd 
            hPos = self.prev_hPos
            bPos = self.prev_bPos
        else:
            if self.parent.fi == 1: bPos = copy(cpt)
            ### motion detection
            m_diff = cv2.absdiff( frame_arr, self.last_motion_frame) # difference between the current and last motion frame 
            if self.motionTh[0] <= np.sqrt(np.sum(m_diff)/255) < self.motionTh[1]: # thresholds for accepting motion
            # there's a motion
                self.last_motion_frame = frame_arr.copy()
                ### determine hPos, bPos, and hD 
                if hPos == None or bPos == None: # it's not manually given by user
                    if animalSp == 'Marmoset': hPos, bPos, hD = self.proc_marmoset(cnt_info)
                    elif animalSp == 'Alligator': hPos, bPos, hD, frame_arr = self.proc_alligator(frame_arr, diff, cntsBr)
                    elif animalSp == 'Gerbil': hPos, bPos, hD, frame_arr = self.proc_alligator(frame_arr, diff, cntsBr) # gerbil can use alligator algorithm, with different parameters (determined in onChangeAnimalSp function of hc.py)

            else: # no motion detection
                if self.parent.fi > 1: # not the first frame
                    hD = self.parent.oData[self.parent.fi-1]["hD"] # set head direction as the previous head direction
                    if type(hD) == int:
                        bPos = self.prev_bPos 
                        hPos = self.calc_hPos(hD, bPos) # calculate hPos
     
        if imgType == 'Greyscale(Diff)': frame_arr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        elif imgType == 'Greyscale(Edge)': frame_arr = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
        if type(hD) == int: # head direction available
            cv2.line(frame_arr, hPos, bPos, (0,255,0), 2)
            cv2.circle(frame_arr, hPos, 3, (0,125,255), -1)
       
        status_msg += 'HD %s'%str(hD)
        
        cv2.putText(frame_arr, status_msg, (10,25), cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(0,250,0), thickness=2) # write status
        self.video_rec.write( cv2.resize(frame_arr,self.video_fSize) )

        if hPos==None: hPos=('None','None')
        if bPos==None: bPos=('None','None')
        if hD==None: hD='None'
        return frame_arr, hPos, bPos, hD

    # --------------------------------------------------
    
    def proc_marmoset(self, cnt_info):
        hPos=None; bPos=None; hD=None
        ept1 = None; ept2 = None
        cnt_info = sorted(cnt_info, reverse=True) # sort contours by size (largest first)
        if len(cnt_info) >= 2: # there are more than 2 contours
            ept1 = (cnt_info[0][1], cnt_info[0][2]) # first two (largest) contours
            ept2 = (cnt_info[1][1], cnt_info[1][2]) # first two (largest) contours
            if ept1[0] > ept2[0]: tmp_ = copy(ept2); ept2 = copy(ept1); ept1 = tmp_
        if ept1 != None:
            ang_ = calc_line_angle(ept1, ept2) # angle of a line connecting two ears
            s = np.sin(np.deg2rad(ang_))
            c = np.cos(np.deg2rad(ang_))
            l_ = np.sqrt( (ept1[0]-ept2[0])**2+(ept1[1]-ept2[1])**2 ) / 2 # length of line
            bPos = ( int(ept1[0]+l_*c), int(ept1[1]-l_*s) )
            ### determine two possible head directions
            if -90 <= ang_ <= 90: hD1 = ang_+90; hD2 = ang_-90
            elif 91 <= ang_ <= 180: hD1 = -(180-(ang_+90)%180); hD2 = ang_-90
            elif -91 > ang_ >= -180: hD1 = ang_+90; hD2 = (ang_-90)%180 
            ### determine head direction which is closer to the previous one
            if type(self.prev_hd) == int: # head direction in the previous hD is available
                ang_diff1 = calc_angle_diff(self.prev_hd, hD1)
                ang_diff2 = calc_angle_diff(self.prev_hd, hD2)
                if ang_diff1 <= ang_diff2:
                    if ang_diff1 <= self.degTh: hD = hD1
                    else: hD = self.prev_hd
                elif ang_diff1 > ang_diff2:
                    if ang_diff2 <= self.degTh: hD = hD2
                    else: hD = self.prev_hd
            else:
                hD = hD1
        else: # ear position is not available
            hD = self.prev_hd
            if type(hD) == int: bPos = self.prev_bPos 
        if type(hD) == int: # head direction is available
            if bPos == None: bPos = self.prev_bPos
            hPos = self.calc_hPos(hD, bPos) # calculate hPos
            
        return hPos, bPos, hD
   
    # --------------------------------------------------

    def proc_alligator(self, frame_arr, diff, cntsBr):
        hPos=None; bPos=None; hD=None
        if type(self.prev_hd) == int: 
            ### draw a point with the known head direction from the center of subject        
            s = np.sin(np.deg2rad(self.prev_hd))
            c = np.cos(np.deg2rad(self.prev_hd))
            w_=abs(cntsBr[2]-cntsBr[0]); h_=abs(cntsBr[3]-cntsBr[1])
            l_ = max(w_,h_) # length of line
            hpt_ = ( int(self.prev_bPos[0]+l_*c), int(self.prev_bPos[1]-l_*s) ) # point further away from head
            cv2.circle(frame_arr, hpt_, 5, (50,50,50), -1)
            ### cluster subject points 
            dPts = np.where(diff==255)
            dPts = np.hstack( (dPts[1].reshape((dPts[1].shape[0],1)),dPts[0].reshape((dPts[0].shape[0],1))) ) # coordinates of all the 255 pixels of diff
            centroids, __ = kmeans( dPts, self.nKMC ) # kmeans clustering
            ### calculate distances between the hpt_ and centroids of clusters. the closest cluster is supposed to be the head cluster
            d_cents = [] # (distance to hpt_, x, y)
            for ci in range(len(centroids)):
                d_ =  np.sqrt((centroids[ci][0]-hpt_[0])**2+(centroids[ci][1]-hpt_[1])**2)
                d_cents.append( [d_, centroids[ci][0], centroids[ci][1]] )
            d_cents = sorted(d_cents) # sort by distance
            centroids = np.array(d_cents, dtype=np.uint16)[:,1:]
            ### get points of the head cluster                            
            idx, __ = vq( dPts, np.asarray(centroids) )
            pts_ = dPts[np.where(idx==0)[0]]
            #frame_arr[pts_[:,1],pts_[:,0]] = self.cluster_cols[0]
            ### determine hPos as the closest point in the head cluster
            d_ = []
            for hci in range(len(pts_)):
                d_.append(  np.sqrt((pts_[hci][0]-hpt_[0])**2+(pts_[hci][1]-hpt_[1])**2) )
            i_ = d_.index(min(d_))
            hPos = ( int(pts_[i_][0]), int(pts_[i_][1]) )
            cv2.circle(frame_arr, hPos, 3, (200,200,200), -1)
            ### draw each cluster centroids and calculates distances between the head cluster and other clusters 
            d_ = [] # distances between the head cluster and other clusters
            for ci in range(1, len(centroids)):
                cv2.circle(frame_arr, tuple(centroids[ci]), 3, (200,200,200), -1)
                d_.append( np.sqrt((centroids[0][0]-int(centroids[ci][0]))**2+(centroids[0][1]-int(centroids[ci][1]))**2) )
                pts_ = dPts[np.where(idx==ci)[0]]
                frame_arr[pts_[:,1],pts_[:,0]] = self.cluster_cols[ci]
            bi_ = d_.index(min(d_))+1 # index of the closets cluster to the head cluster
            bPos = ( int(centroids[bi_][0]), int(centroids[bi_][1]) ) 
            #cv2.circle(frame_arr, hPos, 5, self.cluster_cols[0], -1)
            #cv2.circle(frame_arr, bPos, 5, self.cluster_cols[1], -1)
            #cv2.line(frame_arr, hPos, bPos, (200,200,200), 2)
        if hPos == None and bPos == None:
            hD = self.prev_hd 
        else:
            hD = calc_line_angle(bPos, hPos)
            if self.parent.fi > 1: # not the first frame
                if type(self.prev_hd) == int:
                    if calc_angle_diff(self.prev_hd, hD) > self.degTh: hD = self.prev_hd # differnce is too big. use the previous hD
        if type(hD) == int and bPos != None: hPos = self.calc_hPos(hD, bPos) # calculate hPos with hD
        return hPos, bPos, hD, frame_arr

    # --------------------------------------------------

    def proc_alligator2(self, frame_arr, diff, cntsBr):
        ''' Difference between proc_alligator and proc_alligator2 is that 
        bPos is the red tag in the middle of head
        '''
        hPos=None; bPos=None; hD=None
        if type(self.prev_hd) == int: 
            ### draw a point with the known head direction from the center of subject        
            s = np.sin(np.deg2rad(self.prev_hd))
            c = np.cos(np.deg2rad(self.prev_hd))
            w_=abs(cntsBr[2]-cntsBr[0]); h_=abs(cntsBr[3]-cntsBr[1])
            l_ = max(w_,h_) # length of line
            hpt_ = ( int(self.prev_bPos[0]+l_*c), int(self.prev_bPos[1]-l_*s) ) # point further away from head
            cv2.circle(frame_arr, hpt_, 5, (50,50,50), -1)
            ### cluster subject points 
            dPts = np.where(diff==255)
            dPts = np.hstack( (dPts[1].reshape((dPts[1].shape[0],1)),dPts[0].reshape((dPts[0].shape[0],1))) ) # coordinates of all the 255 pixels of diff
            centroids, __ = kmeans( dPts, self.nKMC ) # kmeans clustering
            ### calculate distances between the hpt_ and centroids of clusters. the closest cluster is supposed to be the head cluster
            d_cents = [] # (distance to hpt_, x, y)
            for ci in range(len(centroids)):
                d_ =  np.sqrt((centroids[ci][0]-hpt_[0])**2+(centroids[ci][1]-hpt_[1])**2)
                d_cents.append( [d_, centroids[ci][0], centroids[ci][1]] )
            d_cents = sorted(d_cents) # sort by distance
            centroids = np.array(d_cents, dtype=np.uint16)[:,1:]
            ### get points of the head cluster                            
            idx, __ = vq( dPts, np.asarray(centroids) )
            pts_ = dPts[np.where(idx==0)[0]]
            #frame_arr[pts_[:,1],pts_[:,0]] = self.cluster_cols[0]
            ### determine hPos as the closest point in the head cluster
            d_ = []
            for hci in range(len(pts_)):
                d_.append(  np.sqrt((pts_[hci][0]-hpt_[0])**2+(pts_[hci][1]-hpt_[1])**2) )
            i_ = d_.index(min(d_))
            hPos = ( int(pts_[i_][0]), int(pts_[i_][1]) )
            cv2.circle(frame_arr, hPos, 3, (200,200,200), -1)
            ### find red tag in the middle head
            tmp_grey_img = self.find_color((0,0,frame_arr.shape[1],frame_arr.shape[0]), frame_arr, (175,100,90), (180,255,255))
            M = cv2.moments(tmp_grey_img)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            bPos = (cx, cy)
        if hPos == None and bPos == None:
            hD = self.prev_hd 
        else:
            hD = calc_line_angle(bPos, hPos)
            if self.parent.fi > 1: # not the first frame
                if type(self.prev_hd) == int:
                    if calc_angle_diff(self.prev_hd, hD) > self.degTh: hD = self.prev_hd # differnce is too big. use the previous hD
        if type(hD) == int and bPos != None: hPos = self.calc_hPos(hD, bPos) # calculate hPos with hD
        return hPos, bPos, hD, frame_arr

    # --------------------------------------------------
    
    def calc_hPos(self, hD, bPos):
        s = np.sin(np.deg2rad(hD))
        c = np.cos(np.deg2rad(hD))
        l_ = self.hdLineLen # length of line
        hPos = ( int(bPos[0]+l_*c), int(bPos[1]-l_*s) )
        return hPos

    # --------------------------------------------------

    def calc_bPos(self, hD, hPos):
        fhD = hD-180 # flip head direction
        if fhD <= -180: fhD += 360 
        s = np.sin(np.deg2rad(fhD))
        c = np.cos(np.deg2rad(fhD))
        l_ = self.hdLineLen # length of line
        bPos = ( int(hPos[0]+l_*c), int(hPos[1]-l_*s) )
        return bPos

    # --------------------------------------------------
    
    def find_color(self, rect, inImage, HSV_min, HSV_max):
    # Find a color(range: 'HSV_min' ~ 'HSV_max') in an area('rect') of an image('inImage')
    # 'rect' here is (x1,y1,x2,y2)
        pts_ = [ (rect[0], rect[1]), (rect[0], rect[3]), (rect[2], rect[3]), (rect[2], rect[1]) ] # Upper Left, Lower Left, Lower Right, Upper Right
        mask = np.zeros( (inImage.shape[0], inImage.shape[1]) , dtype=np.uint8 )
        tmp_grey_img = np.zeros( (inImage.shape[0], inImage.shape[1]) , dtype=np.uint8 )
        cv2.fillConvexPoly(mask, np.asarray(pts_), 255)
        tmp_col_img = cv2.bitwise_and(inImage, inImage, mask=mask )
        HSV_img = cv2.cvtColor(tmp_col_img, cv2.COLOR_BGR2HSV)
        tmp_grey_img = cv2.inRange(HSV_img, HSV_min, HSV_max)
        ret, tmp_grey_img = cv2.threshold(tmp_grey_img, 50, 255, cv2.THRESH_BINARY)
        return tmp_grey_img

    # --------------------------------------------------

    def clustering(self, pt_list, threshold):
        pt_arr = np.asarray(pt_list)
        result = []
        try: result = list(fclusterdata(pt_arr, threshold, 'distance'))
        except: pass
        number_of_groups = 0
        groups = []
        if result != []:
            groups = []
            number_of_groups = max(result)
            for i in range(number_of_groups): groups.append([])
            for i in range(len(result)):
                 groups[result[i]-1].append(pt_list[i])
        return number_of_groups, groups


# ======================================================

if __name__ == '__main__':
    pass

