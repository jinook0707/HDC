from time import time, sleep
from copy import copy
from glob import glob
from random import randint, choice

import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans 
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import pdist # Pairwise distances 
  # between observations in n-dimensional space. 

from fFuncNClasses import get_time_stamp, writeFile
from fFuncNClasses import calc_pt_line_dist, calc_line_angle, calc_angle_diff
from fFuncNClasses import calc_pt_w_angle_n_dist
from fFuncNClasses import load_img, rot_pt, getColorInfo

DEBUG = False 

#=======================================================================

class CVProc:
    """ Class for processing a frame image using computer vision algorithms
    to code animal position/direction/behaviour
        
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, parent):
        if DEBUG: print("CVProc.__init__()")
        
        ##### [begin] setting up attributes -----
        self.p = parent
        self.bg = None # background image of chosen video
        self.fSize = (960, 540) # default frame size
        self.cluster_cols = [
                                (0,0,0), 
                                (255,0,0), 
                                (0,255,0), 
                                (0,0,255), 
                                (255,100,100), 
                                (100,255,100), 
                                (100,100,255), 
                                (0,127,255),
                                (0,255,255), 
                                (127,127,127), 
                            ] # BGR color for each cluster in clustering
        #self.storage = {} # storage for previsouly calculated parameters 
        #  or temporary frame image sotrage, etc...
        ##### [end] setting up attributes -----

    #-------------------------------------------------------------------
    
    def proc_img(self, frame_arr, animalECase, 
                 x, flagMHPos=False, imgType='RGB-image'):
        """ Process frame image to code animal position/direction/behaviour
        
        Args:
            frame_arr (numpy.ndarray): Frame image array.
            animalECase (str): Animal experiment case.
            x (dict): temporary data to process such as 
              hD, bD, hPos, bPos, etc..
            flagMHPos (bool): Whether input of hPos and bPos was give manually. 
            imgType (str): Image type to return.
        
        Returns:
            x (dict): return data, including hD, bD, hPos, bPos, etc..
            frame_arr (numpy.ndarray): Frame image to return after processing.
        """
        if DEBUG: print("CVProc.proc_img()") 
        
        p = self.p # parent
        ecp = self.p.aecParam
        diff = None
        edged = None

        ec_with_bg = ['Marmoset04']
        # chosen video's background image is missing (when required)
        isBGMissing = False 
        if animalECase in ec_with_bg: # background image is required
            if type(self.bg) != np.ndarray: isBGMissing = True 
 
        ##### [begin] calculate data of the current frame ---
        d = p.oData[p.vRW.fi] # output data of the current frame
          # this might have already calculated data
        if flagMHPos:
        # positions of head and the first body seg. are given 
        # (by click & drag)
            # calculate head direction
            x["hD"] = calc_line_angle((x["bPosX"],x["bPosY"]), 
                                      (x["hPosX"],x["hPosY"])) 
            if 'bD' in x.keys(): # body direction
            # Currently, user interface doesn't allow body direction 
            # to be manually marked and given to this function.
                x["bD"] = d[p.bdi]
                if not x["bD"] in ['None', 'D']:
                # valid (integer) body direction is available
                    x["bD"] = int(x["bD"])
                    ### The 2nd and 3rd body seg. positions
                    ###   should be available. copy.
                    x["b1PosX"] = d[p.b1xi]
                    x["b1PosY"] = d[p.b1yi]
                    if not x["b1PosX"] in ['None', 'D']:
                        x["b1PosX"] = int(x["b1PosX"])
                        x["b1PosY"] = int(x["b1PosY"])
                    x["b2PosX"] = d[p.b2xi]
                    x["b2PosY"] = d[p.b2yi]
                    if not x["b2PosX"] in ['None', 'D']:
                        x["b2PosX"] = int(x["b2PosX"])
                        x["b2PosY"] = int(x["b2PosY"])
                else:
                    ### copy previous body direction data
                    x["bD"] = x["p_bD"]
                    x["b1PosX"] = x["p_b1PosX"]
                    x["b1PosY"] = x["p_b1PosY"]
                    x["b2PosX"] = x["p_b2PosX"]
                    x["b2PosY"] = x["p_b2PosY"]
        
        elif x["mHD"] == "True":
        # head direction in this frame was already manually marked before
            pass
        
        elif p.flagContManualInput:
        # continuous manual input is checked
            ### copy data from the previous frame data
            for k in x.keys():
                if k.startswith("p_"): continue
                pk = "p_" + k
                x[k] = x[pk] 
        
        else:
        # else
            if p.vRW.fi > 0:
                ### motion detection
                ###   with difference between the current and last motion frame
                m_diff = cv2.absdiff(frame_arr, self.last_motion_frame)
                m_diff = cv2.cvtColor(m_diff, cv2.COLOR_BGR2GRAY)
                m_val = np.sqrt(np.sum(m_diff)/255)
                m_val_min, m_val_max = ecp["motionTh"]["value"]
            if (p.vRW.fi == 0) or (m_val_min <= m_val < m_val_max):
            # 1st frame or motion detected
                self.last_motion_frame = frame_arr.copy()
                ### process the current frame, using computer vision algorithms
                if animalECase == 'Marmoset04':
                    if not isBGMissing: 
                        x, diff = self.proc_marmoset04(x, frame_arr)
                elif animalECase == 'Macaque19':
                    x, frame_arr, diff = self.proc_macaque19(x, frame_arr)
                elif animalECase == 'Rat05':
                    x, diff, frame_arr = self.proc_rat05(x, frame_arr)
                '''
                elif animalECase == 'Dove19':
                    x = self.proc_dove19(frame_arr, diff)
                    frame_arr = ret["retImg"]
                    diff = ret["retDiff"]
                '''
            else: # no motion detection
                if p.vRW.fi > 0: # not the first frame
                    ### copy data from the previous frame data
                    for k in x.keys():
                        if k.startswith("p_"): continue
                        pk = "p_" + k
                        x[k] = x[pk] 
        ##### [end] calculate data of the current frame ---
     
        if imgType == 'Greyscale(Diff)' and type(diff) == np.ndarray:
            frame_arr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        #elif imgType == 'Greyscale(Edge)' and edged != None:
        #   frame_arr = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

        ### draw head position and direction 
        if type(x["hD"]) == int: # head direction available
            if p.ratFImgDispImg != None:
                r = 1.0/p.ratFImgDispImg
                lw = int(2 * r)
                cr = int(4 * r)
            pt1 = (x["hPosX"], x["hPosY"])
            pt2 = (x["bPosX"], x["bPosY"])
            cv2.line(frame_arr, pt1, pt2, (0,255,0), lw)
            cv2.circle(frame_arr, pt1, cr, (0,125,255), -1)
       
        ### draw status message such as frame-index, head position, etc..
        status_msg = "%i/ %i"%(p.vRW.fi, p.vRW.nFrames-1)
        status_msg += ", HD %s"%(str(x["hD"]))
        if 'bD' in p.dataCols:
            status_msg += ", BD %s"%(str(x["bD"]))
        cv2.putText(frame_arr, 
                    status_msg, 
                    (10,25), 
                    cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=1.0, 
                    color=(0,250,0), 
                    thickness=2)

        if isBGMissing: 
        # chosen video's background image is missing (when required)
            msg = ["This algorithm requires background image.",
                   "Background file, [video-file-name]_bg.jpg,",
                   " is NOT found."]
            ty = 100
            for i in range(len(msg)):
                cv2.putText(frame_arr, 
                            msg[i], 
                            (10, ty), 
                            cv2.FONT_HERSHEY_DUPLEX, 
                            fontScale=1.0, 
                            color=(0,250,0), 
                            thickness=2)
                ty += 50

        if p.flagVRec:
            # write a frame of analysis video recording
            self.p.vRW.writeFrame(frame_arr)

        return x, frame_arr 
   
    #-------------------------------------------------------------------
    
    def proc_marmoset04(self, x, frame_arr):
        """ Calculate head direction of a common marmoset monkey
        
        Args:
            x (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.
        
        Returns:
            x (dict): received 'x' dictionary, but with calculated data.
            diff (numpy.ndarray): grey image after background subtraction.
        """
        if DEBUG: print("CVProc.proc_marmoset04()")

        diffCol, diff = self.procBGSubtraction(frame_arr, self.bg)
        edged = self.getEdged(diff)
        cnt_info, cnt_pts, cnt_br, cnt_cpt = self.getCntData(edged)
       
        # center points of left and right ear
        lEar = [-1, -1]; rEar = [-1, -1] 
        # sort contours by size (largest first)
        cnt_info = sorted(cnt_info, reverse=True)
        if len(cnt_info) >= 2:
        # there are more than 2 contours
            ### consider two largest contours as contours of ears
            lEar = (cnt_info[0][1], cnt_info[0][2]) 
            rEar = (cnt_info[1][1], cnt_info[1][2])
            ### determine left/right ear
            # left ear means it's positioned closer to left side of screen
            if lEar[0] > rEar[0]:
                tmp = copy(rEar)
                rEar = copy(lEar)
                lEar = tmp

        if lEar != [-1, -1]:
        # contours considered to be ears are found
            # calculate angle of a line connecting two ears
            e_ang = calc_line_angle(lEar, rEar)
            
            ### calculate bPos as a middle point between two ear contours
            s = np.sin(np.deg2rad(e_ang))
            c = np.cos(np.deg2rad(e_ang))
            # calculate length of line
            lL = np.sqrt( (lEar[0]-rEar[0])**2+(lEar[1]-rEar[1])**2 ) / 2
            x["bPosX"] = int(lEar[0]+lL*c)
            x["bPosY"] = int(lEar[1]-lL*s)

            ### determine two possible head directions
            if -90 <= e_ang <= 90: 
                hD1 = e_ang+90
                hD2 = e_ang-90
            elif 91 <= e_ang <= 180:
                hD1 = -(360-(e_ang+90))
                hD2 = e_ang-90
            elif -91 > e_ang >= -180:
                hD1 = e_ang+90
                hD2 = (e_ang-90)%180 
            
            ### determine head direction which is closer to the previous one
            if type(x["p_hD"]) == int:
            # head direction in the previous hD is available
                ang_diff1 = calc_angle_diff(x["p_hD"], hD1)
                ang_diff2 = calc_angle_diff(x["p_hD"], hD2)
                if ang_diff1 <= ang_diff2:
                # hD1 is closer to the previous head direction
                    if ang_diff1 <= self.p.aecParam["uDegTh"]["value"]:
                    # head direction change in this frame
                    # is in tolerable difference range
                        x["hD"] = hD1
                    else:
                    # else, keep the head direction from the previous frame
                        x["hD"] = x["p_hD"]
                elif ang_diff1 > ang_diff2:
                # hD2 is closer to the previous head direction
                    if ang_diff2 <= self.p.aecParam["uDegTh"]["value"]:
                        x["hD"] = hD2
                    else:
                        x["hD"] = x["p_hD"]
            else:
            # no previous frame available
                x["hD"] = hD1
        else:
        # ear position is not available
            if self.p.vRW.fi > 0:
                # keep the head direction from previous frame
                x["hD"] = x["p_hD"] 
                if type(x["hD"]) == int:
                    x["bPosX"] = x["p_bPosX"] 
                    x["bPosY"] = x["p_bPosY"]
         
        if type(x["hD"]) == int: # hD is available
            if x["bPosX"] == 'None' and x["p_bPosX"] != 'None': 
            # bPos is not determined and it's available in previous frame
                x["bPosX"] = x["p_bPosX"]
                x["bPosY"] = x["p_bPosY"]
            if type(x["bPosX"]) == int:
                x["hPosX"], x["hPosY"] = calc_pt_w_angle_n_dist(
                                        x["hD"], 
                                        self.p.aecParam["hdLineLen"]["value"],
                                        x["bPosX"],
                                        x["bPosY"],
                                        True,
                                        ) # calculate hPos
        return x, diff

    #-------------------------------------------------------------------
    
    def proc_macaque19(self, x, frame_arr):
        """ Calculate head direction of a macaque monkey
        
        Args:
            x (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.

        Returns:
            x (dict): received 'x' dictionary, but with calculated data.
            frame_arr (numpy.ndarray): Frame image array.
            diff (numpy.ndarray): grey image to show some processing results.
        """
        if DEBUG: print("CVProc.proc_macaque19()") 

        fSh = frame_arr.shape

        ### determine state of the computer screen,
        ###   which can change color of macaque face
        flagScreen = False # bottom of screen is visible in recorded video.
          # before session starts, it's white. (H: 38-105, S: 0-20)
          # after session starts, it alternates between two colors.
          # the secondary color, which briefly turns on time to time,
          #   changes macaque's face color from pinkish to purplish.
          # Non-affecting color is [H: 110-117, S: 98-108]
          # Affecting color is [H: 116-119, S: 46-52]
        pt = (5, int(fSh[0]/2))
        m = 5 # margin around 'pt'
        colInfo = getColorInfo(frame_arr, pt=pt, m=m)
        hm = colInfo["hue_med"]
        sm = colInfo["sat_med"]
        if 95 < hm < 135 and 30 < sm < 65:
            flagScreen = True # face color will be changed
        ### draw to denote where screen color sample was taken
        cv2.rectangle(frame_arr, 
                      (pt[0]-m, pt[1]-m), 
                      (pt[0]+m, pt[1]+m), 
                      (100,100,100), 
                      2)

        rect = (0, 0, fSh[1], fSh[0]) # rect for searching colors
        
        ### find approximate blueish wooden panel area
        colMin = tuple(self.p.aecParam["uCol0Min"]["value"])
        colMax = tuple(self.p.aecParam["uCol0Max"]["value"])
        fcRslt = self.find_color(rect, frame_arr, colMin, colMax)
        edged = self.getEdged(fcRslt)
        cnt_info, cnt_pts, cnt_br, cnt_cpt = self.getCntData(edged)
        ### update rect as the panel area
        bpx1 = cnt_br[0]+cnt_br[2] - int(fSh[0]*0.65)
        bpy1 = cnt_cpt[1] - int(fSh[0]/4)
        bpx2 = cnt_br[0]+cnt_br[2]
        bpy2 = cnt_cpt[1] + int(fSh[0]/4)
        rect = (bpx1, bpy1, bpx2, bpy2)
        # draw the found area
        cv2.rectangle(frame_arr, (bpx1,bpy1), (bpx2,bpy2), (150,150,150), 3)
        
        ### find hair color of head
        if flagScreen:
        # screen color changed to a color that changes macaque's head color
            colMin = tuple(self.p.aecParam["uCol2Min"]["value"])
            colMax = tuple(self.p.aecParam["uCol2Max"]["value"])
            fcRslt_h = self.find_color(rect, frame_arr, colMin, colMax)
        else:
        # normal color 
            colMin = tuple(self.p.aecParam["uCol1Min"]["value"])
            colMax = tuple(self.p.aecParam["uCol1Max"]["value"])
            fcRslt_h = self.find_color(rect, frame_arr, colMin, colMax)
        M = cv2.moments(fcRslt_h)
        if M['m00'] > 0: 
            bx = int(M['m10']/M['m00'])
            by = int(M['m01']/M['m00'])
            x["bPosX"] = bx
            x["bPosY"] = by
            ## update rect as approximate head area
            r = self.p.aecParam["uHRSz"]["value"]/2 
            bx1 = int(bx - fSh[0]*r)
            by1 = int(by - fSh[0]*r)
            bx2 = int(bx + fSh[0]*r)
            by2 = int(by + fSh[0]*r)
            rect = (bx1, by1, bx2, by2)
            # draw the found area
            cv2.rectangle(frame_arr, (bx1,by1), (bx2,by2), (200,200,200), 3)
            
            ### find face color (pinkish-reddish/ purplish)
            # face color is in its normal color 
            colMin = tuple(self.p.aecParam["uCol3Min"]["value"])
            colMax = tuple(self.p.aecParam["uCol3Max"]["value"])
            fcRslt = self.find_color(rect, frame_arr, colMin, colMax)
            if flagScreen:
            # screen color changed to a color that changes macaque's face color
                colMin = tuple(self.p.aecParam["uCol4Min"]["value"])
                colMax = tuple(self.p.aecParam["uCol4Max"]["value"])
                fcr = self.find_color(rect, frame_arr, colMin, colMax)
                # add the secondary color result on the normal color result 
                fcRslt = cv2.add(fcRslt, fcr)
            M = cv2.moments(fcRslt)
            if M['m00'] > 0:
                x["hPosX"] = int(M['m10']/M['m00'])
                x["hPosY"] = int(M['m01']/M['m00'])
                
                if type(x["hPosX"]) == int and type(x["bPosX"]) == int:
                    # calculate head direction
                    x["hD"] = calc_line_angle((x["bPosX"],x["bPosY"]), 
                                               (x["hPosX"],x["hPosY"]))
         
        if self.p.vRW.fi > 1 and x["hD"] == "None":
        # not the 1st frame and head direction was not calculated
            ### copy data of the previous frame
            x["hD"] = x["p_hD"]
            x["hPosX"] = x["p_hPosX"]
            x["hPosY"] = x["p_hPosY"]
            x["bPosX"] = x["p_bPosX"]
            x["bPosY"] = x["p_bPosY"]
         
        return x, frame_arr, fcRslt

    #-------------------------------------------------------------------

    def proc_rat05(self, x, frame_arr):
        """ Calculate head direction of a rat 
        
        Args:
            x (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.

        Returns:
            x (dict): received 'x' dictionary, but with calculated data.
            diff (numpy.ndarray): grey image after background subtraction.
            frame_arr (numpy.ndarray): Frame image array.
        """
        if DEBUG: print("CVProc.proc_rat05()")

        diffCol, diff = self.procBGSubtraction(frame_arr, self.bg)
        edged = self.getEdged(diff)
        cnt_info, cnt_pts, cnt_br, cnt_cpt = self.getCntData(edged)

        if type(x["p_hD"]) == int: 
            ### draw a far point with the known head direction 
            ###   from previous base point (bPos) 
            s = np.sin(np.deg2rad(x["p_hD"]))
            c = np.cos(np.deg2rad(x["p_hD"]))
            lL = max(cnt_br[2], cnt_br[3]) # length of line
            fpt = (int(x["p_bPosX"]+lL*c),
                   int(x["p_bPosY"]-lL*s)) # far front point away from head
            cv2.circle(frame_arr, fpt, 5, (50,50,50), -1)
            ### cluster subject points 
            dPts = np.where(diff==255)
            # coordinates of all the 255 pixels of diff
            dPts = np.hstack((dPts[1].reshape((dPts[1].shape[0],1)),
                              dPts[0].reshape((dPts[0].shape[0],1)))) 
            t_dPts = dPts.astype(np.float32)
            centroids, __ = kmeans(
                                obs=t_dPts,
                                k_or_guess=self.p.aecParam["uNKMC"]["value"]
                                ) # kmeans clustering
            ### calculate distances between the fpt and centroids of clusters,
            ### the closest cluster is supposed to be the head cluster
            d_cents = [] # (distance to fpt, x, y)
            for ci in range(len(centroids)):
                cx, cy = centroids[ci]
                dist = np.sqrt((cx-fpt[0])**2 + (cy-fpt[1])**2)
                d_cents.append([dist, cx, cy])
            d_cents = sorted(d_cents) # sort by distance
            centroids = np.array(d_cents, dtype=np.uint16)[:,1:]
            ### get points of the head cluster                            
            idx, __ = vq(dPts, np.asarray(centroids))
            t_pts = dPts[np.where(idx==0)[0]]
            #frame_arr[t_pts[:,1],t_pts[:,0]] = self.cluster_cols[0]
            ### determine hPos as the closest pixel toward fpt 
            dists = []
            for hci in range(len(t_pts)):
                ptx, pty = t_pts[hci]
                dists.append(np.sqrt((ptx-fpt[0])**2 + (pty-fpt[1])**2))
            idx = dists.index(min(dists))
            ### store hPos
            x["hPosX"] = int(t_pts[idx][0])
            x["hPosY"] = int(t_pts[idx][1])
            col = (200, 200, 200)
            cv2.circle(frame_arr, (x["hPosX"],x["hPosY"]), 3, col, -1)
            ### draw each cluster centroids and 
            ###   calculates distances between 
            ###   the head cluster and other clusters 
            dists = [] # distances between the head cluster and other clusters
            hcx, hcy = centroids[0] # head cluster x & y
            for ci in range(1, len(centroids)):
                cv2.circle(frame_arr, tuple(centroids[ci]), 3, col, -1)
                cx, cy = centroids[ci]
                dists.append(np.sqrt((hcx-int(cx))**2 + (hcy-int(cy))**2))
                t_pts = dPts[np.where(idx==ci)[0]]
                frame_arr[t_pts[:,1],t_pts[:,0]] = self.cluster_cols[ci]
            # index of the closets cluster to the head cluster
            bi = dists.index(min(dists)) + 1
            ### store bPos
            x["bPosX"] = int(centroids[bi][0])
            x["bPosY"] = int(centroids[bi][1])
        if x["hPosX"] == 'None' or x["bPosX"] == 'None':
            x["hD"] = x["p_hD"] 
        else:
            x["hD"] = calc_line_angle((x["bPosX"],x["bPosY"]), 
                                 (x["hPosX"],x["hPosY"]))
            if self.p.vRW.fi > 1: # not the first frame
                if type(x["p_hD"]) == int:
                    degDiffTol = self.p.aecParam["uDegTh"]["value"]
                    if calc_angle_diff(x["p_hD"], x["hD"]) > degDiffTol:
                    # differnce is too big. keep the previous hD
                        x["hD"] = x["p_hD"]
        """ 
        if type(x["hD"]) == int and x["bPosX"] != None:
            hPos = calc_pt_w_angle_n_dist(x["hD"],
                                          self.p.aecParam["hdLineLen"]["value"],
                                          x["bPosX"],
                                          x["bPosY"],
                                          True)
        """
        return x, diff, frame_arr

    '''
    #-------------------------------------------------------------------
    
    def proc_dove19(self, x, frame_arr, diff):
        if DEBUG: print("CVProc.proc_dove19()")
        bPos = None; hD = None; bD = None
        pHP = x["p_hPos"]; pBP = x["p_bPos"]; pB1P = x["p_b1Pos"]; pB2P = x["p_b2Pos"]; pD = x["p_hD"]; pBD = x["p_bD"]

        ### get dove blob rect, using blob size 
        minArea = self.p.aecParam["uBlobSzMin"]["value"] # approximate min. size of dove blob 
        maxArea = self.p.aecParam["uBlobSzMax"]["value"] # approximate max. size of dove blob 
        ccOutput = cv2.connectedComponentsWithStats(diff, connectivity=4) # 0: number_of_labels, 1: labeled image, 2: stat matrix(left, top, width, height, area)
        stats = list(ccOutput[2])
        s = -1 
        for idx in range(1, ccOutput[0]):
            if minArea <= stats[idx][4] <= maxArea:
                s = stats[idx]
                break
        if type(s) == int: return dict(hPos=pHP, bPos=pBP, b1Pos=pB1P, b2Pos=pB2P, hD=pD, bD=pBD, retImg=frame_arr, retDiff=diff)
        doveR = [s[0], s[1], s[0]+s[2], s[1]+s[3]] # get doveR; rect (x1,y1,x2,y2) on overall dove's body 
        diff[ccOutput[1]!=idx] = 0

        ### remove debris, noise, details, fill small holes.. 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) 
        diff = cv2.erode(diff, kernel, 1)
        diff = cv2.dilate(diff, kernel, 1)
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=1)
        for i in range(3): diff = cv2.medianBlur(diff, 5)

        ### get rid of background from color image
        frame = frame_arr.copy()
        frame[diff!=255] = (0, 0, 0) 

        ### get contours
        edged = cv2.Canny(diff, 100, 100)
        (_cntImg, cnts, hierarchy)= cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cntPts = []
        for ci in range(len(cnts)):
            #br = cv2.boundingRect(cnts[ci])
            #if br[2]+br[3] < 20: continue
            cntPts += list( cnts[ci].reshape((cnts[ci].shape[0], 2)) ) 

        ### calculate body line
        def calcLRPts(rr): 
            box = cv2.boxPoints(rr)
            box = np.int0(box)
            ### group 2 closer points
            s1=[]; s2=[]
            _d = []
            _d.append( np.sqrt( (box[0][0]-box[1][0])**2 + (box[0][1]-box[1][1])**2 ) )
            _d.append( np.sqrt( (box[0][0]-box[2][0])**2 + (box[0][1]-box[2][1])**2 ) )
            _d.append( np.sqrt( (box[0][0]-box[3][0])**2 + (box[0][1]-box[3][1])**2 ) )
            iC20 = _d.index(min(_d)) + 1 # index of closest point to the first box point
            s2Idx = range(1,4); s2Idx.remove(iC20)
            s1.append(tuple(box[0])); s1.append(tuple(box[iC20]))
            for s2i in s2Idx: s2.append(tuple(box[s2i]))
            ### get center point of each group and calc. left and right side points
            s1x = int(s1[0][0]+s1[1][0])/2
            s1y = int(s1[0][1]+s1[1][1])/2
            s2x = int(s2[0][0]+s2[1][0])/2
            s2y = int(s2[0][1]+s2[1][1])/2
            if s1x < s2x: lx=s1x; ly=s1y; lpts=s1; rx=s2x; ry=s2y; rpts=s2
            else: lx=s2x; ly=s2y; lpts=s2; rx=s1x; ry=s1y; rpts=s1
            return lx, ly, lpts, rx, ry, rpts
        rr = cv2.minAreaRect(np.asarray(cntPts))
        lx, ly, lpts, rx, ry, rpts = calcLRPts(rr)
        lineCt= (int(rr[0][0]), int(rr[0][1])) # center of the min. area rect

        ### calculate transverse line: 90-deg (counter clockwise) rotated line of the body line
        lx2, ly2 = rot_pt( (lx,ly), lineCt, 90 ) # in rot_pt - 0:right, 90:up, 180:left, 270:down
        rx2, ry2 = rot_pt( (rx,ry), lineCt, 90 )
        if lx2 > rx2:
            _x = int(rx2); _y = int(ry2)
            rx2 = int(lx2); ry2 = int(ly2)
            lx2 = _x; ly2 = _y 

        def getEllipseMaskedImg(ctPt, radius, angle, startAngle, endAngle, shape, img):
            mask = np.zeros( shape, dtype=np.uint8 )
            cv2.ellipse( mask, ctPt, axes=(radius,radius), angle=angle, startAngle=startAngle, endAngle=endAngle, color=255, thickness=-1 )
            # startAngle & endAngle in cv2.ellipse - 0:right, 90:down, 180:left, 270:up
            mask[mask==255] = 1
            if len(img.shape) == 3: mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
            _df = img * mask
            return _df
        
        startAngle = [0, 180]
        endAngle = [180, 360]
        
        ### deterine head side 
        if pHP == ('None', 'None'):
            tLnAng = calc_line_angle((lx2,ly2), (rx2,ry2))
            if tLnAng >= 0: tLnAng = 360 - tLnAng
            else: tLnAng = abs(tLnAng)
            radius = int( np.sqrt( (lineCt[0]-lx)**2 + (lineCt[1]-ly)**2 ) )
            psum = []
            for i in range(len(startAngle)):
                _df = getEllipseMaskedImg(lineCt, radius, tLnAng, startAngle[i], endAngle[i], frame.shape[:2], diff)
                psum.append(np.sum(_df))
            mi = psum.index(max(psum)) # index of degree range(startAngle-endAngle), produced higher sum of pixels
            rotAng = 360-(endAngle[mi]+startAngle[mi])/2 # note: cv2.ellipse - clockwise, rot_pt - counter clockwise  
            hspt = rot_pt( (rx2,ry2), lineCt, rotAng) # head side point
        else:
            if np.sqrt( (lx-pHP[0])**2 + (ly-pHP[1])**2 ) < np.sqrt( (rx-pHP[0])**2 + (ry-pHP[1])**2 ): hspt = (lx, ly)
            else: hspt = (rx, ry)
            if hspt == rot_pt( (rx2,ry2), lineCt, 270): mi = 0
            else: mi = 1
      
        ### re-calculate body line with segmented body part 
        _dl = np.sqrt( (hspt[0]-lx)**2 + (hspt[1]-ly)**2 )
        _dr = np.sqrt( (hspt[0]-rx)**2 + (hspt[1]-ry)**2 )
        if _dl < _dr:
            ang = calc_line_angle((lx,ly), (rx,ry))
            pts = lpts
        else:
            ang = calc_line_angle((rx,ry), (lx,ly))
            pts = rpts
        s = np.sin(np.deg2rad(ang))
        c = np.cos(np.deg2rad(ang))
        l = int(np.sqrt( (lx-rx)**2 + (ly-ry)**2 )) / 5
        pt1=list(pts[0]); pt2=list(pts[1])
        bodyLinePts = []
        ns = 5 # number body segments
        ### line through middle of multiple body segments
        for i in range(ns):
            mask = np.zeros( tuple(frame.shape[:2]) , dtype=np.uint8 )
            pt3 = [ int(pt2[0]+l*c), int(pt2[1]-l*s) ]
            pt4 = [ int(pt1[0]+l*c), int(pt1[1]-l*s) ]
            if i == 1: s2ct = ( int(np.average([pt1[0],pt2[0],pt3[0],pt4[0]])), int(np.average([pt1[1],pt2[1],pt3[1],pt4[1]])) ) # store the 2nd segment's center point
            cv2.fillPoly(mask, [np.asarray([pt1,pt2,pt3,pt4], np.int32)], 255) 
            pt1=list(pt3); pt2=list(pt4)
            if i == 0: tl = [ tuple(pt3), tuple(pt4) ]
            _diff = diff.copy()
            _diff[mask==0] = 0
            if i == 0: # head segment
                _r = float(np.count_nonzero(_diff))/np.count_nonzero(mask) # this ratio is high when head is tucked in body area, while low when head is out of body
                ### determine the method to calculate head direction, depending on this ratio 
                if _r < self.p.aecParam["uHDCalcTh"]["value"]:
                    hDMethod = "gradualDiffDec"
                else:
                    hDMethod = "" 
            M = cv2.moments(_diff)
            if M['m00'] == 0:
                return dict(hPos=pHP, bPos=pBP, b1Pos=pB1P, b2Pos=pB2P, hD=pD, bD=pBD, retImg=frame_arr, retDiff=diff)
            mbx = int(M['m10']/M['m00'])
            mby = int(M['m01']/M['m00'])
            bodyLinePts.append( (mbx, mby) )
        hspt = tuple(bodyLinePts[1])
        if ang > 0: _ang = ang-180 # flip angle
        else: _ang = 180-abs(ang)
        s = np.sin(np.deg2rad(_ang))
        c = np.cos(np.deg2rad(_ang))
        l /= 2
        hspt = ( int(hspt[0]+l*c), int(hspt[1]-l*s) )
        bl = [hspt, tuple(bodyLinePts[ns-2])]
        lineCt = ( (bl[0][0]+bl[1][0])/2, (bl[0][1]+bl[1][1])/2 ) # center of body line
        if tl[0][0] > tl[1][0]:
            _x = int(tl[1][0]); _y = int(tl[1][1])
            tl[1] = (int(tl[0][0]), int(tl[0][1]))
            tl[0] = (_x, _y)
        tLnAng = calc_line_angle(tl[0], tl[1]) # re-calculate angle of transverse line
        if tLnAng >= 0: tLnAng = 360 - tLnAng
        else: tLnAng = abs(tLnAng)
        maxRad = int( np.sqrt( (tl[0][0]-tl[1][0])**2 + (tl[0][1]-tl[1][1])**2 ) / 2 ) 
      
        ###### beginning of calculating hPos and bPos
        if hDMethod == "gradualDiffDec":
        # determine hPos, bPos with gradually decreasing half circles
            ### remove part of 'diff' & 'frame' to leave only head side area
            mask = np.zeros( tuple(frame.shape[:2]) , dtype=np.uint8 )
            diff2 = diff.copy()
            for i in range(len(startAngle)):
                if i == mi: continue 
                r = frame.shape[0]/2
                #cv2.ellipse( mask, s2ct, axes=(r,r), angle=tLnAng, startAngle=startAngle[i], endAngle=endAngle[i], color=255, thickness=-1 )
                #frame[mask==255] = (0, 0, 0)
                cv2.ellipse( mask, hspt, axes=(r,r), angle=tLnAng, startAngle=startAngle[i], endAngle=endAngle[i], color=255, thickness=-1 )
                diff2[mask==255] = 0
            for i in range(2):
                if i == 0 : bPos = hspt; continue # testing hspt instead of calculation for bPos
                if i == 0: th = self.p.aecParam["uSzTh4PH"]["value"]
                elif i == 1: th = self.p.aecParam["uSzTh4DH"]["value"]
                prevSum = -1
                for r in range(int(maxRad*1.25), int(maxRad*0.1), -5):
                # gradually decrease head side diff image
                # when decrement is under threshold, break the loop. this will leave protruding part of head-segment, which would approximately leave dove's head(i==0) and beak(i==1).
                    tmpDiff = getEllipseMaskedImg(hspt, r, tLnAng, startAngle[mi], endAngle[mi], frame.shape[:2], diff2) 
                    s = np.sum(tmpDiff)/255
                    dec = s-prevSum
                    if prevSum != -1 and dec < -th: break
                    prevSum = copy(s)
                _diff = diff2.copy()
                _diff[tmpDiff==255] = 0
                if i == 0: pDiff = _diff.copy()
                elif i == 1: dDiff = _diff.copy()
                ### calculate potential bPos(i==0) & hPos(i==1)
                M = cv2.moments(_diff)
                if M['m00'] == 0: 
                    return dict(hPos=pHP, bPos=pBP, b1Pos=pB1P, b2Pos=pB2P, hD=pD, bD=pBD, retImg=frame_arr, retDiff=diff)
                pt = ( int(M['m10']/M['m00']), int(M['m01']/M['m00']) )
                if i == 0: bPos = tuple(pt) 
                elif i == 1: hPos = tuple(pt) 
        else:
        # head is tucked in body            
            ###### determine bPos
            grey = cv2.cvtColor(frame_arr, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(grey)
            tRad = int(maxRad*1.25)
            cv2.circle(mask, hspt, tRad, 255, -1)
            frame[mask!=255] = (0, 0, 0) 
            grey[mask!=255] = 0
            grey = cv2.medianBlur(grey, 7)
            ### make an image to subtract to remove outer body line 
            bLEdge = cv2.Canny(grey, 120, 250) 
            (_cntImg, cnts, hierarchy) = cv2.findContours(bLEdge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # get contours 
            cv2.drawContours(bLEdge, cnts, -1, 255, 5) # draw thick body line 
            eImg = cv2.Canny(grey, 30, 60) # edge detection
            eImg = cv2.subtract(eImg, bLEdge) # remove body line
            ### remove line of circluar image cut
            mask.fill(0)
            cv2.circle(mask, hspt, tRad, 255, 10)
            eImg[mask==255] = 0
            ### find head by finding a circle
            hc = None
            for th in range(10, 30): # gradually increase threshold
                c = cv2.HoughCircles(eImg, cv2.HOUGH_GRADIENT, 1.0, 20, param1=50, param2=th, minRadius=int(maxRad*0.25), maxRadius=int(maxRad*0.5)) # find circle
                if c is None: continue
                c = c[0,:]
                if len(c) > 1: continue # more than one circle was found. thershold is too low.
                hc = c[0]
                break
            
            if not hc is None:
                cv2.circle(frame_arr, (hc[0], hc[1]), hc[2], (255,255,255), 2)
                bPos = ( int(hc[0]), int(hc[1]) )
                ###### determine hPos
                bRad = int(maxRad*0.25)
                sRad = int(hc[2]+bRad)
                x = bPos[0]+sRad; y = bPos[1]
                rslt = []
                testing = False 
                if testing:
                    gc = grey.copy()
                    gc2 = np.zeros_like(gc) 
                grey = cv2.medianBlur(grey, 9)
                eImg = cv2.Canny(grey, 30, 100)
                ### remove line of circluar image cut
                mask.fill(0)
                cv2.circle(mask, hspt, tRad, 255, 5)
                eImg[mask==255] = 0
                for i in range(2):
                    sa = 30 - (i*10) # step angle
                    if i > 0: x += int(bRad*1.5)
                    for a in range(0, 360, sa):
                        _x, _y = rot_pt( (x,y), bPos, a) # in rot_pt - 0:right, 90:up, 180:left, 270:down
                        mask.fill(0)
                        cv2.circle(mask, (_x,_y), bRad, 255, -1)
                        if testing: cv2.circle(gc, (_x,_y), bRad, 255, 1)
                        _e = eImg.copy()
                        _e[mask!=255] = 0
                        if testing: gc2 = cv2.add(gc2, _e) 
                        M = cv2.moments(_e)
                        rslt.append( [M['m00'], (_x,_y)] )
                if testing:
                    cv2.imwrite('x0_gc.jpg', gc)
                    cv2.imwrite('x0_gc2.jpg', gc2)
                rslt = sorted(rslt, reverse=True)
                hPos = rslt[0][1]
            else:
                bPos = pBP; hPos = pHP
            ######--------------
        ###### end of calculating hPos and bPos

        b1Pos = bl[0]; b2Pos = bl[1] 
        if hPos != None and bPos != None and type(hPos[0]) == int and type(hPos[1]) == int: 
            distBH = int(np.sqrt( (bPos[0]-hPos[0])**2 + (bPos[1]-hPos[1])**2 ))
            hD = calc_line_angle(bPos, hPos) # calculate head direction
            bD = calc_line_angle(b2Pos, b1Pos) # calculate body direction
            hPos = calc_pt_w_angle_n_dist(
                            hD, 
                            self.p.aecParam["hdLineLen"]["value"],
                            bPos[0],
                            bPos[1],
                            True
                            ) # re-calculate hPos (for uniform length 
                              #   of the head direction line)
            if (type(pD)==int and calc_angle_diff(pD,hD)>self.p.aecParam["uDegTh"]["value"]) or (distBH < self.p.aecParam["uBHDistMin"]["value"]):
            # if angle differnce is too big or current bPos and hPos are too close, use the previous hD
                hD = pD 
                hPos = pHP
                bPos = pBP
        else:
            hD = pD 
            hPos = pHP
            bPos = pBP
                
        ### draw to the resultant frame image 
        cv2.rectangle(frame_arr, (doveR[0], doveR[1]), (doveR[2], doveR[3]), (200,200,200), 1) # rectangle around dove
        frame_arr = cv2.add(frame_arr, cv2.cvtColor(edged,cv2.COLOR_GRAY2BGR)) # draw recognized edge of dove
        for i in range(1, len(bodyLinePts)):
            cv2.line(frame_arr, bodyLinePts[i-1], bodyLinePts[i], (100,100,150), 30) # draw segmented body lines
        cv2.line(frame_arr, bl[0], bl[1], (0,0,255), 2) # draw line along the whole body
        cv2.line(frame_arr, tl[0], tl[1], (255,255,255), 2) # draw transverse line of the body

        if hDMethod == "gradualDiffDec":
            #frame_arr[pDiff==255] = (50,200,200) # larger head area for bPos
            frame_arr[dDiff==255] = (50,50,200) # distal part of the head area 

        #frame_arr = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
        return dict(hPos=hPos, bPos=bPos, b1Pos=b1Pos, b2Pos=b2Pos, hD=hD, bD=bD, retImg=frame_arr, retDiff=diff)
    ''' 
    #-------------------------------------------------------------------
    
    def getYForFittingLine(self, pts, br, lx, rx, width):
        ''' get left and right side y-coordinates to draw fitting line
        '''
        if DEBUG: print("CVProc.getYForFittingLine()")
        (vx, vy, x, y) = cv2.fitLine(pts, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01) # DIST_L2: least square distance p(r) = r^2/2
        yunit = float(vy)/vx
        ly = int(-x*yunit + y) # left-most y-corrdinate
        ry = int((width-x)*yunit + y) # right-most y-coordinate
        if lx != 0: ly = int(lx*yunit + ly)
        if rx != width: ry = int((rx-lx)*yunit + ly)
        ### remove parts of the fitting line, which go out of overall bounding rect
        if ly < br[1]:
            rat = float(br[1]-ly)/(ry-ly)
            lx = lx + int((rx-lx)*rat)
            ly = br[1]
        elif ly > br[1]+br[3]:
            rat = float((br[1]+br[3])-ly)/(ry-ly)
            lx = lx + int((rx-lx)*rat)
            ly = br[1]+br[3]
        if ry < br[1]:
            rat = float(br[1]-ly)/(ry-ly)
            rx = lx + int((rx-lx)*rat)
            ry = br[1]
        elif ry > br[1]+br[3]:
            rat = float((br[1]+br[3])-ly)/(ry-ly)
            rx = lx + int((rx-lx)*rat)
            ry = br[1]+br[3] 
        return int(ly), int(ry), yunit
    
    #-------------------------------------------------------------------

    def calc_bPos(self, hD, hPos):
        if DEBUG: print("CVProc.calc_bPos()")
        fhD = hD-180 # flip head direction
        if fhD <= -180: fhD += 360 
        s = np.sin(np.deg2rad(fhD))
        c = np.cos(np.deg2rad(fhD))
        l_ = self.p.aecParam["hdLineLen"]["value"] # length of line
        bPos = ( int(hPos[0]+l_*c), int(hPos[1]-l_*s) )
        return bPos

    #-------------------------------------------------------------------
    
    def find_color(self, rect, inImage, HSV_min, HSV_max):
    # Find a color(range: 'HSV_min' ~ 'HSV_max') in an area('rect') of an image('inImage')
    # 'rect' here is (x1,y1,x2,y2)
        if DEBUG: print("CVProc.find_color()")
        pts_ = [ (rect[0], rect[1]), (rect[0], rect[3]), (rect[2], rect[3]), (rect[2], rect[1]) ] # Upper Left, Lower Left, Lower Right, Upper Right
        mask = np.zeros( (inImage.shape[0], inImage.shape[1]) , dtype=np.uint8 )
        tmp_grey_img = np.zeros( (inImage.shape[0], inImage.shape[1]) , dtype=np.uint8 )
        cv2.fillConvexPoly(mask, np.asarray(pts_), 255)
        tmp_col_img = cv2.bitwise_and(inImage, inImage, mask=mask )
        HSV_img = cv2.cvtColor(tmp_col_img, cv2.COLOR_BGR2HSV)
        tmp_grey_img = cv2.inRange(HSV_img, HSV_min, HSV_max)
        ret, tmp_grey_img = cv2.threshold(tmp_grey_img, 50, 255, cv2.THRESH_BINARY)
        return tmp_grey_img

    #-------------------------------------------------------------------

    def clustering(self, pt_list, threshold):
        if DEBUG: print("CVProc.clustering()")
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

    #-------------------------------------------------------------------

    def procBGSubtraction(self, img, bgImg):
        """ Get some informative images after subtracting background.

        Args:
            img (numpy.ndarray): image array to process.
            bgImg (numpy.ndarray): background image to subtract.

        Returns:
            diff (numpy.ndarray): greyscale image after BG subtraction.
            edged (numpy.ndarray): greyscale image of edges in 'diff'.
        """
        if DEBUG: print("CVProc.procBGSubtraction()")
        
        ### get difference between
        ###   the current frame and the background image 
        diffCol = cv2.absdiff(img, bgImg)
        diff = cv2.cvtColor(diffCol, cv2.COLOR_BGR2GRAY)
        ecp = self.p.aecParam
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        if "bgsMExOIter" in ecp.keys() and ecp["bgsMExOIter"]["value"] != -1:
            diff = cv2.morphologyEx(
                        diff, 
                        cv2.MORPH_OPEN, 
                        kernel, 
                        iterations=ecp["bgsMExOIter"]["value"],
                        ) # to decrease noise & minor features
        if "bgsMExCIter" in ecp.keys() and ecp["bgsMExCIter"]["value"] != -1:
            diff = cv2.morphologyEx(
                        diff, 
                        cv2.MORPH_CLOSE, 
                        kernel, 
                        iterations=ecp["bgsMExCIter"]["value"],
                        ) # closing small holes
        if "bgsThres" in ecp.keys() and ecp["bgsThres"]["value"] != -1:
            __, diff = cv2.threshold(
                            diff, 
                            ecp["bgsThres"]["value"], 
                            255, 
                            cv2.THRESH_BINARY
                            ) # make the recognized part clear 
        return diffCol, diff
    
    #-------------------------------------------------------------------
    
    def getEdged(self, greyImg):
        """ Find edges of greyImg

        Args:
            greyImg (numpy.ndarray): greyscale image to extract edges.

        Returns:
            (numpy.ndarray): greyscale image with edges.
        """
        if DEBUG: print("CVProc.getEdged()")

        return cv2.Canny(greyImg,
                         self.p.aecParam["cannyTh"]["value"][0],
                         self.p.aecParam["cannyTh"]["value"][1])
    
    #-------------------------------------------------------------------
    
    def getCntData(self, img):
        """ Get some useful data from contours in a given image.

        Args:
            img (numpy.ndarray): greyscale image to get contour data.

        Returns:
            cnt_info (list): contour info list. 
                each item is a tuple (size, center-X, center-Y) of a contour.
                'size' is width + height.
            cnt_pts (list): list of every pixels in all contours.
            cnt_cpt (tuple): center point of contour.
        """
        if DEBUG: print("CVProc.getCntData()")

        # find contours
        (cnts, hierarchy) = cv2.findContours(img, 
                                             cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
        cnt_info = [] # contour info (size, center-X, center-Y)
        cnt_pts = [] # put points of all contours into this list
        for ci in range(len(cnts)):
            mr = cv2.boundingRect(cnts[ci])
            if mr[2]+mr[3] < self.p.aecParam["contourTh"]["value"]: continue
            #cv2.circle(img, (mr[0]+mr[2]/2,mr[1]+mr[3]/2), mr[2]/2, 125, 1)
            cnt_info.append((mr[2]+mr[3], mr[0]+mr[2]/2, mr[1]+mr[3]/2))
            cnt_pts += list(cnts[ci].reshape((cnts[ci].shape[0], 2)))
        if len(cnt_pts) > 0:
            # rect bounding all contour points
            cnt_br = cv2.boundingRect(np.asarray(cnt_pts))
            # calculate center point of all contours
            cnt_cpt = (cnt_br[0]+int(cnt_br[2]/2), cnt_br[1]+int(cnt_br[3]/2))
        else:
            cnt_br = (-1, -1, -1, -1)
            cnt_cpt = (-1, -1)

        return cnt_info, cnt_pts, cnt_br, cnt_cpt
    
    #-------------------------------------------------------------------

#=======================================================================

if __name__ == '__main__':
    pass

