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

from datetime import datetime
from time import time, sleep
from glob import glob
from os import path, getcwd
from subprocess import Popen, PIPE
from random import randint
import shlex, Queue

import wx
import numpy as np
import cv2

# --------------------------------------------

def GNU_notice(idx=0):
    '''
      function for printing GNU copyright statements
    '''
    if idx == 0:
        print '''
CATOS Copyright (c) 2015 Jinook Oh, W. Tecumseh Fitch.
This program comes with ABSOLUTELY NO WARRANTY; for details run this program with the option `-w'.
This is free software, and you are welcome to redistribute it under certain conditions; run this program with the option `-c' for details.
'''
    elif idx == 1:
        print '''
THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.
'''
    elif idx == 2:
        print '''
You can redistribute this program and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
'''

# --------------------------------------------

def get_time_stamp(flag_ms=False):
    ts = datetime.now()
    ts = ('%.4i_%.2i_%.2i_%.2i_%.2i_%.2i')%(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
    if flag_ms == True: ts += '_%.6i'%(ts.microsecond)
    return ts

# --------------------------------------------

def writeFile(file_path, txt, mode='a'):
    f = open(file_path, mode)
    f.write(txt)
    f.close()

# --------------------------------------------

def run_terminal_cmd(cmd):
    if type(cmd) == str: cmd = shlex.split(cmd)
    p = Popen(cmd, stdout=PIPE)
    stdout, stderr = p.communicate()
    return stdout, stderr

# --------------------------------------------

def chk_msg_q(msg_q):
    msg=''; msg_src=''; msg_body=''; msg_details=''
    if msg_q.empty() == False:
        msg = msg_q.get(False)
        msg = msg.split('/')
        msg_src = msg[0]; msg_body = msg[1]
        if len(msg) > 2: msg_details = msg[2:]
    return msg_src, msg_body, msg_details

# --------------------------------------------

def calc_line_angle(pt1, pt2):
    return int( np.degrees( np.arctan2(-(pt2[1]-pt1[1]),pt2[0]-pt1[0]) ) ) # 0=right, 90=upward, -90=downward, 180=left

# --------------------------------------------
def calc_angle_diff(ang1, ang2):
    ''' calculate angle difference between two angles
    '''
    angle_diff = 0
    if (ang1 >= 0 and ang2 >= 0) or (ang1 < 0 and ang2 < 0): angle_diff = abs(ang1-ang2)
    elif (ang1 >=0 and ang2 < 0) or (ang1 < 0 and ang2 >= 0):
        ad1 = abs(ang1) + abs(ang2)
        ad2 = 180-abs(ang1) + 180-abs(ang2)
        angle_diff = min(ad1, ad2)
    return angle_diff

# --------------------------------------------

def calc_pt_line_dist(pt, line, flag_line_ends=True):
    ''' calculates distance froma a point to a line
    pt : point (x0,y0)
    line : line ((x1,y1), (x2,y2))
    flag_line_ends : line ends at (x1,y1) & (x2,y2)
    '''
    lpt1 = line[0]; lpt2 = line[1]
    ldx = lpt2[0]-lpt1[0]
    ldy = lpt2[1]-lpt1[1]
    sq_llen = ldx**2 + ldy**2 # square length of line
    if sq_llen == 0: # line is a point
        return np.sqrt( (pt[0]-lpt1[0])**2 + (pt[1]-lpt1[1])**2 )
    u = ( (pt[0]-lpt1[0])*ldx + (pt[1]-lpt1[1])*ldy ) / float(sq_llen)
    x = lpt1[0] + u * ldx
    y = lpt1[1] + u * ldy
    if flag_line_ends:
        if u < 0.0: x, y = lpt1[0], lpt1[1] # beyond lpt1-end of segment
        elif u > 1.0: x, y = lpt2[0], lpt2[1] # beyond lpt2-end of segment
    dx = pt[0] - x
    dy = pt[1] - y
    return np.sqrt(dx**2 + dy**2)

# --------------------------------------------

def make_b_curve_coord(orig, h1, h2, dest, num_of_track_pts=100, accel=False):
    ''' returns a list of points to make a Bezier curve.
    accel = True ; point moves faster as it goes along the curve
    '''
    if accel == True:
        track_idx = []
        for i in xrange(num_of_track_pts): track_idx.append( int(1.1 ** i) )
        num_of_track_pts = int( 1.1 ** (num_of_track_pts-1) )
    else:
        track_idx = range(num_of_track_pts)
    p0p1X_step = (h1[0] - orig[0]) / float(num_of_track_pts)
    p0p1Y_step = (h1[1] - orig[1]) / float(num_of_track_pts)
    if h2 == None:
        p1p2X_step = (dest[0] - h1[0]) / float(num_of_track_pts)
        p1p2Y_step = (dest[1] - h1[1]) / float(num_of_track_pts)
    else:
        p1p2X_step = (h2[0] - h1[0]) / float(num_of_track_pts)
        p1p2Y_step = (h2[1] - h1[1]) / float(num_of_track_pts)
        p2p3X_step = (dest[0] - h2[0]) / float(num_of_track_pts)
        p2p3Y_step = (dest[1] - h2[1]) / float(num_of_track_pts)
    track = []
    for i in track_idx:
        p0p1X = orig[0] + p0p1X_step * i # q0 x
        p0p1Y = orig[1] + p0p1Y_step * i # q0 y
        p1p2X = h1[0] + p1p2X_step * i # q1 x
        p1p2Y = h1[1] + p1p2Y_step * i # q1 y
        if h2 != None:
            p2p3X = h2[0] + p2p3X_step * i # q2 x
            p2p3Y = h2[1] + p2p3Y_step * i # q2 y
        q0q1X_step = (p1p2X - p0p1X) / float(num_of_track_pts)
        q0q1Y_step = (p1p2Y - p0p1Y) / float(num_of_track_pts)
        if h2 != None:
            q1q2X_step = (p2p3X - p1p2X) / float(num_of_track_pts)
            q1q2Y_step = (p2p3Y - p1p2Y) / float(num_of_track_pts)
        q0q1X = p0p1X + q0q1X_step * i # r0 x
        q0q1Y = p0p1Y + q0q1Y_step * i # r0 y
        if h2 == None:
            track_pt = (q0q1X, q0q1Y)
        else:
            q1q2X = p1p2X + q1q2X_step * i # r1 x
            q1q2Y = p1p2Y + q1q2Y_step * i # r1 y
            r0r1X_step = (q1q2X - q0q1X) / float(num_of_track_pts)
            r0r1Y_step = (q1q2Y - q0q1Y) / float(num_of_track_pts)
            r0r1X = int(q0q1X + r0r1X_step * i)
            r0r1Y = int(q0q1Y + r0r1Y_step * i)
            track_pt = (r0r1X, r0r1Y)
        track.append(track_pt)
    return track

# --------------------------------------------    

def chk_resource_usage(program_log_path):
    ### Check & logging overall cpu & memory usage
    cmd = "top -l 1"
    stdout, stderr = run_terminal_cmd(cmd)
    values = stdout.split("\n")
    log_msg = 'Resource Usage Check @ %s\n===========================\n---(Overall)-------------\n'%get_time_stamp()
    log_msg += '%s\n%s\n%s'%(values[3], values[6], values[7])
    writeFile(program_log_path, log_msg)

    ### Check & logging AA program's processes' resource usage
    colNames = ["command", "pid", "%cpu", "%mem", "vsize"]
    value_list = []
    base_cmd = ["ps", "-a"]
    for colName in colNames:
        cmd = base_cmd + ["-o", colName]
        stdout, stderr = run_terminal_cmd(cmd)
        values = stdout.split("\n")[1:]
        values = [token.strip() for token in values if token != '']
        value_list.append(values)

    my_result_idx = []
    for i in range(len(value_list[0])):
        filename = value_list[0][i].split(" ")[-1]
        if filename.startswith('catos.'): my_result_idx.append(i) # append the index, if this is the process caused by catos program
    ### write header
    log_msg = '---(CATOS\'s processes)-------------\n'
    for col_idx in range(1, len(colNames)): log_msg += colNames[col_idx] + ', '
    log_msg = log_msg.rstrip(', ')
    log_msg += "\n----------------"
    writeFile(program_log_path, log_msg)
    ### write resource-usage
    for i in range(len(my_result_idx)):
        log_msg = ''
        for col_idx in range(1, len(colNames)):
            log_msg += value_list[col_idx][my_result_idx[i]] + ', '
        log_msg = log_msg.rstrip(', ')
        writeFile(program_log_path, log_msg)
    writeFile(program_log_path, "===========================\n")

# --------------------------------------------

def show_msg(msg, parent=None, pos=None, size=(400,150), cancel_btn=False):
    if pos == None: pos = (wx.GetDisplaySize()[0]-1040, 60)
    dlg = PopupDialog(parent, inString=msg, pos=pos, size=size, cancel_btn=cancel_btn)
    result = dlg.ShowModal()
    dlg.Destroy()
    if cancel_btn == False: return True
    elif cancel_btn == True:
        if result == wx.ID_OK: return True
        else: return False

# --------------------------------------------

def load_img(file_path, size=(-1,-1), flag='wx'):
    if flag == 'wx':
        tmp_null_log = wx.LogNull() # for not displaying the tif library warning
        img = wx.Image(file_path, wx.BITMAP_TYPE_ANY)
        del tmp_null_log
        if size != (-1,-1) and type(size[0]) == int and type(size[1]) == int: 
            if img.GetSize() != size: img = img.Rescale(size[0], size[1])
    elif flag == 'cv':
        img = cv2.imread(file_path)
        if size != (-1,-1) and type(size[0]) == int and type(size[1]) == int:
            img = cv2.resize(img, size)
    return img

# --------------------------------------------

def cvImg_to_wxBMP(cvImg):
    cvImg = cv2.cvtColor(cvImg, cv2.cv.CV_BGR2RGB)
    img = wx.EmptyImage(cvImg.shape[1], cvImg.shape[0])
    img.SetData(cvImg.tostring())
    bmp = img.ConvertToBitmap()
    return bmp

# ===========================================================

class PopupDialog(wx.Dialog):
# Class for showing any message to the participant
    def __init__(self, parent = None, id = -1, title = "Message", inString = "", font = None, pos = None, size = (400, 150), cancel_btn = False):
        wx.Dialog.__init__(self, parent, id, title)
        self.SetSize(size)
        if pos == None: self.Center()
        else: self.SetPosition(pos)
        txt = wx.StaticText(self, -1, label = inString, pos = (20, 20))
        txt.SetSize(size)
        if font == None: font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL, False, "Arial", wx.FONTENCODING_SYSTEM)
        txt.SetFont(font)
        txt.Wrap(size[0]-30)
        okButton = wx.Button(self, wx.ID_OK, "OK")
        b_size = okButton.GetSize()
        okButton.SetPosition((size[0] - b_size[0] - 20, size[1] - b_size[1] - 40))
        okButton.SetDefault()
        if cancel_btn == True:
            cancelButton = wx.Button(self, wx.ID_CANCEL, "Cancel")
            b_size = cancelButton.GetSize()
            cancelButton.SetPosition((size[0] - b_size[0]*2 - 40, size[1] - b_size[1] - 40))

#====================================================



