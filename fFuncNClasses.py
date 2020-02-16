# coding: UTF-8
"""
Frequenty used functions and classes

Dependency:
    wxPython (4.0), 
    Numpy (1.17), 
"""

import sys, errno
from os import path, strerror
from datetime import datetime

import wx
import wx.lib.scrolledpanel as sPanel
import numpy as np
import cv2

DEBUG = False

#-----------------------------------------------------------------------

def GNU_notice(idx=0):
    """ Function for printing GNU copyright statements

    Args:
        idx (int): Index to determine which statement to print out.

    Returns:
        None

    Examples:
        >>> GNU_notice(0)
        Copyright (c) ...
        ...
        run this program with option '-c' for details.
    """
    if DEBUG: print("fFuncNClasses.GNU_notice()")

    if idx == 0:
        year = datetime.now().year
        msg = "Copyright (c) %i Jinook Oh, W. Tecumseh Fitch.\n"%(year)
        msg += "This program comes with ABSOLUTELY NO WARRANTY;"
        msg += " for details run this program with the option `-w'."
        msg += "This is free software, and you are welcome to redistribute"
        msg += " it under certain conditions;"
        msg += " run this program with the option `-c' for details."
    elif idx == 1:
        msg = "THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED"
        msg += " BY APPLICABLE LAW. EXCEPT WHEN OTHERWISE STATED IN WRITING"
        msg += " THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE"
        msg += " PROGRAM 'AS IS' WITHOUT WARRANTY OF ANY KIND, EITHER"
        msg += " EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE"
        msg += " IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A"
        msg += " PARTICULAR PURPOSE."
        msg += " THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE"
        msg += " PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU"
        msg += " ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR"
        msg += " CORRECTION."
    elif idx == 2:
        msg = "You can redistribute this program and/or modify it under" 
        msg += " the terms of the GNU General Public License as published"
        msg += " by the Free Software Foundation, either version 3 of the"
        msg += " License, or (at your option) any later version."
    print(msg)

#-----------------------------------------------------------------------

def chkFPath(fp):
    """ Check whether file/folder exists
    If not found, raise FileNotFoundError

    Args:
        fp: file or folder path to check

    Returns:
        None
    
    Examples:
        >>> chkFPath('./test/test1.txt')

    Raises:
       FileNotFoundError: When 'fp' is not a valid file-path. 
    """
    if DEBUG: print("fFuncNClasses.chkFPath()")
    
    rslt = False 
    if path.isdir(fp): rslt = True
    elif path.isfile(fp): rslt = True
    if rslt == False:
        raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), fp)

#-----------------------------------------------------------------------

def get_time_stamp(flag_ms=False):
    """ Function to return string which contains timestamp.

    Args:
        flag_ms (bool, optional): Whether to return microsecond or not

    Returns:
        ts (str): Timestamp string

    Examples:
        >>> print(get_time_stamp())
        2019_09_10_16_21_56
    """
    if DEBUG: print("fFuncNClasses.get_time_stamp()")
    
    ts = datetime.now()
    ts = ('%.4i_%.2i_%.2i_%.2i_%.2i_%.2i')%(ts.year, 
                                            ts.month, 
                                            ts.day, 
                                            ts.hour, 
                                            ts.minute, 
                                            ts.second)
    if flag_ms == True: ts += '_%.6i'%(ts.microsecond)
    return ts

#-----------------------------------------------------------------------

def writeFile(file_path, txt='', mode='a'):
    """ Function to write a text or numpy file.

    Args:
        file_path (str): File path for output file.
        txt (str): Text to print in the file.
        mode (str, optional): File opening mode.

    Returns:
        None

    Examples:
        >>> writeFile('logFile.txt', 'A log is written.', 'a')
    """
    if DEBUG: print("writeFile()")
    
    f = open(file_path, mode)
    f.write(txt)
    f.close()

#-----------------------------------------------------------------------

def str2num(s, c=''):
    """ Function to convert string to an integer or a float number.
    
    Args: 
        s (str): String to process 
        c (str): Intented conversion

    Returns:
        oNum (None/ int/ float):
          Converted number or None (when it failed to convert).

    Examples:
        >>> print(str2num('test'))
        None
        >>> print(str2num('3'))
        3 
        >>> print(str2num('3.0'))
        3.0
        >>> print(str2num('3.0', 'int'))
        3
    """
    if DEBUG: print("fFuncNClasses.str2num()")
    
    oNum = None 
    if c != '': # conversion method is given
        try: oNum = eval('%s(%s)'%(c, s)) # try the intended conversion
        except: pass
    else: # no conversion is specified
        try:
            oNum = int(s) # try to convert to integer first
        except:
            try: oNum = float(s) # then, float
            except: pass
    return oNum 

#-----------------------------------------------------------------------

def load_img(fp, size=(-1,-1), flag='wx'):
    """ Load an image using wxPython or OpenCV functions.

    Args:
        fp (str): File path of an image to load. 
        size (tuple): Output image size.
        flag (str): 'wx' or 'cv'

    Returns:
        img (wx.Image)

    Examples:
        >>> img1 = load_img("test.png")
        >>> img2 = load_img("test.png", size=(300,300))
    """
    if DEBUG: print("fFuncNClasses.load_img()")
    
    chkFPath(fp) # chkeck whether file exists
    
    if flag == 'wx':
        tmp_null_log = wx.LogNull() # for not displaying 
          # the tif library warning
        img = wx.Image(fp, wx.BITMAP_TYPE_ANY)
        del tmp_null_log
        if size != (-1,-1) and type(size[0]) == int and \
          type(size[1]) == int: # appropriate size is given
            if img.GetSize() != size:
                img = img.Rescale(size[0], size[1])
    
    elif flag == 'cv':
        img = cv2.imread(fp)
        if size != (-1,-1) and type(size[0]) == int and type(size[1]) == int:
            img = cv2.resize(img, size)

    return img

#-----------------------------------------------------------------------

def getColorInfo(img, pt, m=1):
    """ Get color information around the given position (pt)

    Args:
        img (numpy.ndarray): Image array
        pt (tuple): x, y coordinate
        m (int): Margin to get area around the 'pt'

    Returns:
        colInfo (dict): Information about color
    """ 
    if DEBUG: print("fFuncNClasses.getColorInfo()")

    r = [pt[0]-m, pt[1]-m, pt[0]+m+1, pt[1]+m+1] # rect
    roi = img[r[1]:r[3],r[0]:r[2]] # region of interest
    col = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    col = col.reshape((col.shape[0]*col.shape[1], col.shape[2]))
    colInfoKey1 = ['hue', 'sat', 'val']
    colInfoKey2 = ['med', 'std']
    colInfo = {}
    for k1i in range(len(colInfoKey1)):
        k1 = colInfoKey1[k1i]
        for k2 in colInfoKey2:
            k = k1 + "_" + k2 
            if k2 == 'med': colInfo[k] = int(np.median(col[:,k1i]))
            elif k2 == 'std': colInfo[k] = int(np.std(col[:,k1i]))
    return colInfo 

#-----------------------------------------------------------------------

def set_img_for_btn(imgPath, btn, imgPCurr=None, imgPDis=None, 
                    imgPFocus=None, imgPPressed=None):
    """ Set image(s) for a wx.Button

    Args:
        imgPath (str): Path of default image file. 
        btn (wx.Button): Button to put image(s).
        imgPCurr (str): Path of image for when mouse is over.
        imgPDis (str): Path of image for when button is disabled.
        imgPFocus (str): Path of image for when button has the keyboard focus.
        imgPPressed (str): Path of image for when button was pressed.

    Returns:
        btn (wx.Button): Button after processing.

    Examples:
        >>> btn = set_img_for_btn('btn1img.png', wx.Button(self, -1, 'testButton'))
    """
    if DEBUG: print("fFuncNClasses.set_img_for_btn()")
    
    imgPaths = dict(all=imgPath, current=imgPCurr, disabled=imgPDis,
                    focus=imgPFocus, pressed=imgPPressed)
    for key in imgPaths.keys():
        fp = imgPaths[key]
        if fp == None: continue
        img = load_img(fp)
        bmp = wx.Bitmap(img)
        if key == 'all': btn.SetBitmap(bmp)
        elif key == 'current': btn.SetBitmapCurrent(bmp)
        elif key == 'disabled': btn.SetBitmapDisabled(bmp)
        elif key == 'focus': btn.SetBitmapFocus(bmp)
        elif key == 'pressed': btn.SetBitmapPressed(bmp)
    return btn

#-----------------------------------------------------------------------

def getWXFonts(initFontSz=8, numFonts=5, fSzInc=2, fontFaceName=""):
    """ For setting up several fonts (wx.Font) with increasing size.

    Args:
        initFontSz (int): Initial (the smallest) font size.
        numFonts (int): Number of fonts to return.
        fSzInc (int): Increment of font size.
        fontFaceName (str, optional): Font face name.

    Returns:
        fonts (list): List of several fonts (wx.Font)

    Examples:
        >>> fonts = getWXFonts(8, 3)
        >>> fonts = getWXFonts(8, 3, 5, 'Arial')
    """
    if DEBUG: print("fFuncNClasses.getWXFonts()")

    if fontFaceName == "":
        if 'darwin' in sys.platform: fontFaceName = "Monaco"
        else: fontFaceName = "Courier"
    fontSz = initFontSz 
    fonts = []  # larger fonts as index gets larger 
    for i in range(numFonts):
        fonts.append(
                        wx.Font(
                                fontSz, 
                                wx.FONTFAMILY_SWISS, 
                                wx.FONTSTYLE_NORMAL, 
                                wx.FONTWEIGHT_BOLD,
                                False, 
                                faceName=fontFaceName,
                               )
                    )
        fontSz += fSzInc 
    return fonts

#-----------------------------------------------------------------------

def setupStaticText(panel, label, name=None, size=None, 
                    wrapWidth=None, font=None, fgColor=None, bgColor=None):
    """ Initialize wx.StatcText widget with more options
    
    Args:
        panel (wx.Panel): Panel to display wx.StaticText.
        label (str): String to show in wx.StaticText.
        name (str, optional): Name of the widget.
        size (tuple, optional): Size of the widget.
        wrapWidth (int, optional): Width for text wrapping.
        font (wx.Font, optional): Font for wx.StaticText.
        fgColor (wx.Colour, optional): Foreground color 
        bgColor (wx.Colour, optional): Background color 

    Returns:
        wx.StaticText: Created wx.StaticText object.

    Examples :
        (where self.panel is a wx.Panel, and self.fonts[2] is a wx.Font object)
        >>> sTxt1 = setupStaticText(self.panel, 'test', font=self.fonts[2])
        >>> sTxt2 = setupStaticText(self.panel, 
                                    'Long text................................',
                                    font=self.fonts[2], 
                                    wrapWidth=100)
    """ 
    if DEBUG: print("fFuncNClasses.setupStaticText()")

    sTxt = wx.StaticText(panel, -1, label)
    if name != None: sTxt.SetName(name)
    if size != None: sTxt.SetSize(size)
    if wrapWidth != None: sTxt.Wrap(wrapWidth)
    if font != None: sTxt.SetFont(font)
    if fgColor != None: sTxt.SetForegroundColour(fgColor) 
    if bgColor != None: sTxt.SetBackgroundColour(bgColor)
    return sTxt

#-----------------------------------------------------------------------

def updateFrameSize(wxFrame, w_sz):
    """ Set window size exactly to a user-defined window size (w_sz)
    , excluding counting menubar/border/etc.

    Args:
        wxFrame (wx.Frame): Frame to resize.
        w_sz (tuple): Client size. 

    Returns:
        None

    Examples:
        >>> updateFrameSize(self, (800,600))
    """
    if DEBUG: print("updateFrameSize()")

    ### set window size to w_sz, excluding counting menubar/border/etc.
    _diff = (wxFrame.GetSize()[0]-wxFrame.GetClientSize()[0], 
             wxFrame.GetSize()[1]-wxFrame.GetClientSize()[1])
    _sz = (w_sz[0]+_diff[0], w_sz[1]+_diff[1])
    wxFrame.SetSize(_sz) 
    wxFrame.Refresh()

#-----------------------------------------------------------------------

def add2gbs(gbs, 
            widget, 
            pos, 
            span=(1,1), 
            bw=5, 
            flag=wx.ALIGN_CENTER_VERTICAL|wx.ALL):
    """ Add 'widget' to given 'gbs'.
    
    Args:
        gbs (wx.GridBagSizer).
        widget (wxPython's widget such as wx.StaticText, wx.Choice, ...).
        pos (tuple): x and y cell indices for positioning 'widget' in 'gbs'.
        span (tuple): width and height in terms of cells in 'gbs'.
        bw (int): Border width.
        flag (int): Flags for styles.
    
    Returns:
        None
    
    Examples:
        >>> add2gbs(self.gbs["ui"], sTxt, (0,0), (1,1))
    """
    if DEBUG: print("fFuncNClasses.add2gbs()")
    
    gbs.Add(widget, pos=pos, span=span, border=bw, flag=flag)

#-----------------------------------------------------------------------

def stopAllTimers(timer):
    """ Stop all running timers
    
    Args:
        timer (dict): container of all timers to stop.
    
    Returns:
        timer (dict) 
    """
    if DEBUG: print("fFuncNClasses.stopAllTimers()")

    for k in timer.keys():
        if timer[k] != None:
            try: timer[k].Stop()
            except: pass
            timer[k] = None
    return timer

#-----------------------------------------------------------------------

def calcI2DIRatio(img, dispSz): 
    """ Calculate ratio for resizing frame image to 
        display image (in StaticBitmap, paintDC, etc)

    Args:
        img (numpy.ndarray): Input image.
        dispSz (tuple): Width and height of StaticBitmap to display image.

    Returns:
        ratImg2DispImg (float): Float number for resizing image later.
    """ 
    if DEBUG: print("fFuncNClasses.calcFI2DIRatio()")

    if img.shape[1] > dispSz[0] or img.shape[0] > dispSz[1]:
        ratImg2DispImg = float(dispSz[0]) / img.shape[1]
        w = img.shape[1]*ratImg2DispImg
        h = img.shape[0]*ratImg2DispImg
        if h > dispSz[1]:
            ratImg2DispImg = float(dispSz[1]) / img.shape[0]
    else:
        ratImg2DispImg = 1.0
    return ratImg2DispImg 

#-----------------------------------------------------------------------

def convt_idx_to_ordinal(number):
    """ Convert zero-based index number to ordinal number string
    0->1st, 1->2nd, ...

    Args:
        number (int): An unsigned integer number.

    Returns:
        number (str): Converted string

    Examples:
        >>> convt_idx_to_ordinal(0)
        '1st'
    """
    if DEBUG: print("fFuncNClasses.convt_idx_to_ordinal()")
    
    if number == 0: return "1st"
    elif number == 1: return "2nd"
    elif number == 2: return "3rd"
    else: return "%ith"%(number+1)

#-----------------------------------------------------------------------

def convt_180_to_360(ang):
    """ convert -180~180 degree to 0~360 degree

    Args:
        angle (int): Output angle, 0 indicates right, 90 indicates up, 
            180 indicates left, -90 indicates down
        
    Returns:
       angle (int): Input angle. 0 indicates right, 90 indicates up,
            180 indicates left, 270 indicates down. 
    
    Examples:
        >>> convt_180_to_360(45)
        45
        >>> convt_180_to_360(-45)
        315
    """ 
    if ang < 0: ang = 360 + ang
    return ang

#-----------------------------------------------------------------------

def convt_360_to_180(angle):
    """ Convert 360 degree system to 180 degree system.

    Args:
        angle (int): Input angle. 0 indicates right, 90 indicates up,
            180 indicates left, 270 indicates down.

    Returns:
        angle (int): Output angle, 0 indicates right, 90 indicates up, 
            180 indicates left, -90 indicates down

    Examples:
        >>> convt_360_to_180(90)
        90
        >>> convt_360_to_180(270)
        -90
    """
    if angle <= 180: return angle
    else: return -(360 % angle)

#-----------------------------------------------------------------------

def calc_pt_w_angle_n_dist(angle, dist, bPosX=0, bPosY=0, flagScreen=False):
    """ Calculates a point when a angle and a distance is given.

    Args:
        angle (int): 0 indicates right, 90 indicates up, 
            180 or -180 indicates left, -90 indicates down.
        dist (int): Distance in pixel.
        bPosX (int): x-coordinate of base-position.
        bPosY (int): y-coordinate of base-position.
        flagScreen (bool): whether it's for displaying it on screen.
          (y coordinate will be reversed)

    Returns:
        (int, int): x,y coordinate

    Examples:
        >>> calc_pt_w_angle_n_dist(90, 20)
        (0, 20)
        >>> calc_pt_w_angle_n_dist(180, 20)
        (-20, 0)
        >>> calc_pt_w_angle_n_dist(-135, 20)
        (-14, -14)
        >>> calc_pt_w_angle_n_dist(-135, 20, 100, 100)
        (85, 85)
        >>> calc_pt_w_angle_n_dist(-135, 20, 100, 100, True)
        (85, 114)
    """
    if DEBUG: print("fFuncNClasses.calc_pt_w_angle_n_dist()")

    s = np.sin(np.deg2rad(angle))
    c = np.cos(np.deg2rad(angle))
    x = int(bPosX + c*dist)
    if flagScreen: y = int(bPosY - s*dist)
    else: y = int(bPosY + s*dist)
    return (x, y)

#-----------------------------------------------------------------------

def calc_line_angle(pt1, pt2):
    """ Calculates angle of a line, defined with two points (pt1 and pt2)

    Args:
        pt1 (tuple): x, y coordinate of the first point
        pt2 (tuple): x, y coordinate of the second point

    Returns:
        (int): angle of the line; 0=right, 90=upward, -90=downward, 180=left

    Examples:
        >>> calc_line_angle((0,0), (1,0))
        0
        >>> calc_line_angle((0,0), (1,1))
        -45
        >>> calc_line_angle((0,0), (-1,-1))
        135
    """
    return int(np.degrees(np.arctan2(-(pt2[1]-pt1[1]),pt2[0]-pt1[0])))

#-----------------------------------------------------------------------

def calc_angle_diff(ang1, ang2):
    """ Calculates angle difference between two angles

    Args:
        ang1 (int): Angle between -180 and 180
        ang2 (int): Angle between -180 and 180

    Returns:
        (int): Angle difference (smallest)

    Examples:
        >>> calc_angle_diff(0, 90)
        90
        >>> calc_angle_diff(180, 45)
        135
        >>> calc_angle_diff(180, -90)
        90
    """
    angle_diff = 0
    if (ang1 >= 0 and ang2 >= 0) or (ang1 < 0 and ang2 < 0):
        angle_diff = abs(ang1-ang2)
    elif (ang1 >=0 and ang2 < 0) or (ang1 < 0 and ang2 >= 0):
        ad1 = abs(ang1) + abs(ang2)
        ad2 = 180-abs(ang1) + 180-abs(ang2)
        angle_diff = min(ad1, ad2)
    return angle_diff

#-----------------------------------------------------------------------

def calc_pt_line_dist(pt, line, flag_line_ends=True):
    """ Calculates distance from a point to a line

    Args:
        pt (tuple): a point
        line (tuple): a line, defined with two points; ((x1,y1), (x2,y2))
        flag_line_ends : whether line ends at (x1,y1) & (x2,y2) 
            or indefinitely extends

    Returns:
        (float): the distance between point and line

    Examples:
        >>> calc_pt_line_dist((0,0), ((1,0), (0,1)))
        0.7071067811865476
        >>> calc_pt_line_dist((0,0), ((1,0), (0.5,0.25)), True)
        0.5590169943749475
        >>> calc_pt_line_dist((0,0), ((1,0), (0.5,0.25)), False)
        0.4472135954999579
    """
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

#-----------------------------------------------------------------------

def rot_pt(pt, ct, deg):
    """ Rotate (counter-clockwise) point;pt, around center point;ct
    * y-coordinate follows computer screen coordinate system,
    where 0 is the top row and the row index increases as it comes down

    Args:
        ct (tuple): Center point
        deg (int): Angel to rotate

    Returns:
        (tuple): Rotated point. 
    
    Examples:
        >>> rot_pt((2,2), (1,1), 45)
        (2, 1)
        >>> rot_pt((2,2), (1,1), 180)
        (0, 0)
        >>> rot_pt((2,2), (1,1), -90)
        (0, 2)
    """ 
    if DEBUG: print("fFuncNClasses.rot_pt()")

    r = np.deg2rad(deg)
    tx = pt[0]-ct[0]
    ty = pt[1]-ct[1]
    x = (tx * np.cos(r) + ty * np.sin(r)) + ct[0]
    y = (-tx * np.sin(r) + ty * np.cos(r)) + ct[1]
    return (int(np.round(x)), int(np.round(y)))

#-----------------------------------------------------------------------

def receiveDataFromQueue(q, logFile=''):
    """ Receive data from a queue.

    Args:
        q (Queue): Queue to receive data.
        logFile (str): File path of log file.

    Returns:
        rData (): Data received from the given queue. 

    Examples:
        >>> receiveDataFromQueue(Queue(), 'log.txt')
    """
    if DEBUG: print("fFuncNClasses.receiveDataFromQueue()")

    rData = None
    try:
        if q.empty() == False: rData = q.get(False)
    except Exception as e:
        em = "%s, [ERROR], %s\n"%(get_time_stamp(), str(e))
        if path.isfile(logFile) == True: writeFile(logFile, em)
        print(em)
    return rData    

#=======================================================================

class PopupDialog(wx.Dialog):
    """ Class for showing a message to a user.
    Most simple messages can be dealt using wx.MessageBox.
    This class was made to use it as a base class for a dialog box
      with more widgets such as a dialog box to enter
      subject's information (id, gender, age, prior experiences, etc)
      before running an experiment.
    
    Args:
        parent (wx.Frame): Parent object (probably, wx.Frame or wx.Panel).
        id (int): ID of this dialog.
        title (str): Title of the dialog.
        msg (str): Message to show.
        iconFP (str): File path of an icon image.
        font (wx.Font): Font of message string.
        pos (None/ tuple): Position to make the dialog window.
        size (tuple): Size of dialog window.
        flagOkayBtn (bool): Whether to show Ok button.
        flagCancelBtn (bool): Whether to show Cancel button.
        flagDefOK (bool): Whether Ok button has focus by default (so that 
          user can just press enter to dismiss the dialog window).
    """
    def __init__(self, 
                 parent=None, 
                 id=-1, 
                 title="Message", 
                 msg="", 
                 iconFP="", 
                 font=None, 
                 pos=None, 
                 size=(300, 200), 
                 flagOkayBtn=True, 
                 flagCancelBtn=False, 
                 flagDefOK=False):
        if DEBUG: print("PopupDialog.__init__()")

        ### init Dialog
        wx.Dialog.__init__(self, parent, id, title)
        self.SetSize(size)
        if pos == None: self.Center()
        else: self.SetPosition(pos)
        self.Center()
        # init panel
        panel = sPanel.ScrolledPanel(self, -1, pos=(0,0), size=size)

        ### font setup 
        if font == None:
            font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.NORMAL, 
                           wx.FONTWEIGHT_NORMAL, False, "Arial", 
                           wx.FONTENCODING_SYSTEM)

        ##### [begin] set up widgets -----
        gbs = wx.GridBagSizer(0,0)
        row = 0; col = 0
        ### icon image
        if iconFP != "" and path.isfile(iconFP) == True:
            bmp = wx.Bitmap(wxLoadImg(iconFP))
            icon_sBmp = wx.StaticBitmap(panel, -1, bmp)
            iconBMPsz = icon_sBmp.GetBitmap().GetSize()
            add2gbs(gbs, icon_sBmp, (row,col), (1,1))
            col += 1 
        else:
            iconFP = ""
            iconBMPsz = (0, 0)
        ### message to show
        sTxt = wx.StaticText(panel, -1, label=msg)
        sTxt.SetSize((size[0]-max(iconBMPsz[0],100)-50, -1))
        sTxt.SetFont(font)
        if iconFP == "": sTxt.Wrap(size[0]-30)
        else: sTxt.Wrap(size[0]-iconBMPsz[0]-30)
        if iconFP == "": _span = (1,2)
        else: _span = _span = (1,1)
        add2gbs(gbs, sTxt, (row,col), _span)
        ### okay button
        row += 1; col = 0
        btn = wx.Button(panel, wx.ID_OK, "OK", size=(100,-1))
        add2gbs(gbs, btn, (row,col), (1,1))
        if flagOkayBtn: # okay button is shown
            if flagCancelBtn == False or flagDefOK == True:
            # cancel button won't be made or default-okay is set True 
                panel.Bind(wx.EVT_KEY_DOWN, self.onKeyPress)
                btn.SetDefault()
        else:
            btn.Hide()
        ### cancel button
        col += 1
        if flagCancelBtn:
            btn = wx.Button(panel, wx.ID_CANCEL, "Cancel", size=(100,-1))
            add2gbs(gbs, btn, (row,col), (1,1))
        else:
            sTxt = wx.StaticText(panel, -1, label=" ")
            add2gbs(gbs, sTxt, (row,col), (1,1))
        ### lay out
        panel.SetSizer(gbs)
        gbs.Layout()
        panel.SetupScrolling()
        ##### [end] set up widgets -----
    
    #-------------------------------------------------------------------

    def onKeyPress(self, event):
        """ Process key-press event
        
        Args: event (wx.Event)
        
        Returns: None
        """
        if DEBUG: print("PopupDialog.onKeyPress()")

        if event.GetKeyCode() == wx.WXK_RETURN: 
            self.EndModal(wx.ID_OK)
    
#=======================================================================

if __name__ == '__main__':
    pass
