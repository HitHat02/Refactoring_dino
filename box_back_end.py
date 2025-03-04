import sys
# from PyQt5 import QtGui
# from PyQt5.QtWidgets import *
# from PyQt5.QtGui import *
# from PyQt5.QtCore import *
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import csv
import time
# import pyqtgraph as pg
# from front_end import *
import math
# np.seterr(divide='ignore', invalid='ignore')
import copy
from scipy import special
# import bitstring

import time

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn
    # pass

class GPRdataHandler():
    def __init__(self, data):
        self.data = data
        # self.readRd3()
        # self.start = self.reshapeRd3()
        self.distanceRatio = 0.075369
        self.chOffsets = [2.58, 2.58, 2.58, 2.58, 2.58, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044,
         0.044, 0.044, 0.044, 0.044, 2.58, 2.58, 2.58, 2.58, 2.58]
        self.data_channel = 25

    # @logging_time
    def readRd3(self):
        gpr = np.frombuffer(self.data, dtype=np.short)
        return gpr

    # @logging_time
    def reshapeRd3(self, gpr):

        gprLen = int(len(gpr))
        gpr_reshaping = gpr.reshape(int(len(gpr) / self.data_channel / 256), self.data_channel, 256)
        # self.gpr_reshaped = np.zeros((25, 256, int(len(self.gpr) / 25 / 256)))
        # for reshape_ch in range(0, 25):
        #     self.gpr_reshaped[reshape_ch] = gpr_reshaping[:, :, :][:, reshape_ch, :].T
        gpr_reshaped = np.swapaxes(np.swapaxes(gpr_reshaping,0,1),1,2)

        return gpr_reshaped

    def alingnSignal(self, gpr_reshaped):
        minimum_list = []
        maximum_list = []

        for align_channel in range(0, self.data_channel):
            ground_avg_list = []
            for align_depth_001 in range(0, 256):
                ground_avg_list.append(np.mean(gpr_reshaped[align_channel][align_depth_001, :]))
            ground_avg_list = np.int32(ground_avg_list)

            for align_depth_002 in range(0, len(ground_avg_list - 1)):
                if ground_avg_list[align_depth_002] < - 1000 and ground_avg_list[align_depth_002 + 1] - ground_avg_list[
                    align_depth_002] > 0:
                    minimum = ground_avg_list[align_depth_002]
                    minimum_idx = align_depth_002
                    minimum_list.append(minimum)
                    break

            for align_depth_003 in range(minimum_idx, len(ground_avg_list - 1)):
                if ground_avg_list[align_depth_003] > 1000 and ground_avg_list[align_depth_003 + 1] - ground_avg_list[
                    align_depth_003] < 0:
                    maximum = ground_avg_list[align_depth_003]
                    maximum_idx = align_depth_003
                    maximum_list.append(maximum)
                    break

        range_list = np.array(maximum_list) - np.array(minimum_list)
        multiple_list = (range_list.max() / range_list)**0.5

        align_test2_mean_mult = np.empty((gpr_reshaped.shape))
        for i in range(gpr_reshaped.shape[1]):
            for j in range(gpr_reshaped.shape[2]):
                align_test2_mean_mult[:, i, j] = gpr_reshaped[:, i, j] * multiple_list

        gpr_reshaped = align_test2_mean_mult
        return gpr_reshaped

    # @logging_time
    def alignGround(self, gpr_reshaped):

        ground_idx_list = [10 for i in range(self.data_channel)]
        minimum_idx = 0
        maximum_idx = 0

        for align_channel in range(0, self.data_channel):
            ground_avg_list = []
            for align_depth_001 in range(0, 256):
                ground_avg_list.append(np.mean(gpr_reshaped[align_channel][align_depth_001, :]))
            ground_avg_list = np.int32(ground_avg_list)

            for align_depth_002 in range(0, len(ground_avg_list - 1)):
                if ground_avg_list[align_depth_002] < - 1000 and ground_avg_list[align_depth_002 + 1] - ground_avg_list[
                    align_depth_002] > 0:
                    minimum = ground_avg_list[align_depth_002]
                    minimum_idx = align_depth_002
                    break

            for align_depth_003 in range(minimum_idx, len(ground_avg_list - 1)):
                if ground_avg_list[align_depth_003] > 1000 and ground_avg_list[align_depth_003 + 1] - ground_avg_list[align_depth_003] < 0:
                    maximum = ground_avg_list[align_depth_003]
                    maximum_idx = align_depth_003
                    break

            for align_depth_004 in range(minimum_idx, maximum_idx + 1):
                if ground_avg_list[align_depth_004] > 0:
                    # VER : 평균과 양수의 평균
                    uint_idx = align_depth_004
                    mean_idx = (minimum_idx + maximum_idx) / 2
                    ground_idx = round((uint_idx + mean_idx) / 2)
                    ground_idx_list[align_channel] = ground_idx
                    break



        gpr_reshaped2 = np.zeros((self.data_channel, 256, len(gpr_reshaped[0][0])))
        for align_channel3 in range(0, self.data_channel):
            gpr_reshaped2[align_channel3, 0:256 - ground_idx_list[align_channel3] + 10, :] = gpr_reshaped[
                                                                                             align_channel3,
                                                                                             ground_idx_list[
                                                                                                 align_channel3] - 10:256,
                                                                                             :]
        return gpr_reshaped2

    # @logging_time
    def alignChannel(self, gpr_reshaped2):

        gpr_aligned = gpr_reshaped2
        chOffsets = np.array(self.chOffsets)
        chOffsets -= np.min(chOffsets)

        for i, value in enumerate(chOffsets):
            if value == 0:
                continue
            gpr_aligned[i, :, int(value/self.distanceRatio):] = gpr_reshaped2[i, :, :-int(value/self.distanceRatio)]
            for align_depth3 in range(0, 256):
                gpr_aligned[i, align_depth3, :int(value/self.distanceRatio)] = gpr_aligned[i, align_depth3, :int(value/self.distanceRatio)] * 0  + int(np.mean(gpr_aligned[i, align_depth3, int(value/self.distanceRatio):]))


        return gpr_aligned

    # @logging_time
    def Backgroud_remove(self, gpr_aligned):
        gpr_AB = gpr_aligned * 0
        for bg_channel in range(0, self.data_channel):
            for bg_depth2 in range(0, 256):
                gpr_AB[bg_channel, bg_depth2] = gpr_aligned[bg_channel, bg_depth2] \
                                                - np.mean(gpr_aligned[bg_channel, bg_depth2],dtype=np.int)

        return gpr_AB

# @logging_time
class drawXYZ():

    def __init__(self, Drawing, depth_maximum, range_value, *box_list):
        # self.Drawing = Drawing
        self.depth_maximum = depth_maximum

        # self.range_vaule_1 = 5000  # default : 5000
        # if self.Drawing.max() > - self.Drawing.min():
        MaxRange = Drawing.max()
        MinRange = Drawing.min()
        fixMax = MaxRange - MinRange

        NoBias = (Drawing - MinRange)

        self.Drawing = (np.divide(NoBias , fixMax, dtype=np.float16) * 255).astype(np.uint8)

        # self.Drawing = self.Drawing.astype(np.uint8)

        self.imgPosition = [0,0,0]
        self.boxPosition = [0,0,0,0,0,0,0,0]

        self.YZ_imgList = []
        self.XY_imgList = []
        self.XZ_imgList = []

        self.yz_draw_check = False
        self.xy_draw_check = False
        self.xz_draw_check = False

        self.stackCheck = False

    def drawYZ(self, i, display_width=-1, screen_position=0):
        if display_width != None and screen_position != None:
            YZ_imgArg = self.Drawing[i, :self.depth_maximum, screen_position:screen_position + display_width]
        else:
            YZ_imgArg = self.Drawing[i, :self.depth_maximum, :]
        return YZ_imgArg

    def drawXY(self, i, display_width=-1, screen_position=0):
        if self.Drawing.ndim == 4:
            Drawing = self.Drawing[...,0:3]
            colored = (Drawing[..., 0] != Drawing[...,1]) | (Drawing[..., 1] != Drawing[...,2]) | (Drawing[..., 0] != Drawing[...,2])
            XY_max = colored.max(axis=1)
            if display_width != None and screen_position != None:
                NpyArea_xy = Drawing[:, i, screen_position:screen_position + display_width].copy()
            else:
                NpyArea_xy = Drawing[:, i, :].copy()

            NpyArea_xy[...,1] = np.where(XY_max>0,XY_max*255*0.7,NpyArea_xy[...,2])
            NpyArea_xy[...,1] = np.where(colored[:, i, :],Drawing[:, i, :, 1],NpyArea_xy[...,1]) #np.float16(NpyArea_xy[...,2]) / np.float16(test)
            NpyArea_xy = np.uint8(NpyArea_xy)
        else:
            if display_width != None and screen_position != None:
                NpyArea_xy = self.Drawing[:, i, screen_position:screen_position + display_width]
            else:
                NpyArea_xy = self.Drawing[:, i, :]
        XY_imgArg = cv2.resize(NpyArea_xy, dsize=(0, 0), fx=1, fy=4, interpolation=cv2.INTER_LINEAR)
        return XY_imgArg

    def drawXZ(self,i):

        NpyArea_xz = np.swapaxes(self.Drawing[:, :self.depth_maximum, i],0,1)
        XZ_imgArg = cv2.resize(NpyArea_xz, dsize=(0, 0), fx=4, fy=1, interpolation=cv2.INTER_LINEAR)
        return XZ_imgArg

    def YZ_showImgByPosition(self, position=[0, 0, 0], invert=False, display_width=None, screen_position=None):

        YZ_imgArg = np.copy(self.drawYZ(position[0], display_width, screen_position))

        # if YZ_imgArg.ndim == 2:
        #     YZ_imgArg = np.stack([YZ_imgArg] * 3,axis=2)
        #
        # elif YZ_imgArg.ndim == 3:
        #     YZ_imgArg = np.concatenate((YZ_imgArg[...,0,np.newaxis], YZ_imgArg[...,1,np.newaxis], YZ_imgArg[...,2,np.newaxis]),axis=2)

        YZ_img_ = YZ_imgArg

        if invert:
            YZ_img_ = 255 - YZ_img_

        YZ_img_ = cv2.cvtColor(YZ_img_, cv2.COLOR_GRAY2RGB)
        h, w, c = YZ_img_.shape

        qImg = QtGui.QImage(YZ_img_.data, w, h, w*c, QtGui.QImage.Format_RGB888)

        return qImg

    def XY_showImgByPosition(self, position=[0, 0, 0], invert=False, display_width=None, screen_position=None):
        XY_imgArg = np.copy(self.drawXY(position[1], display_width, screen_position))
        # if XY_imgArg.ndim == 2:
        #     XY_imgArg= np.stack([XY_imgArg] * 3,axis=2)
        XY_img_ = XY_imgArg

        if invert:
            XY_img_ = 255 - XY_img_

        XY_img_ = cv2.cvtColor(XY_img_, cv2.COLOR_GRAY2RGB)
        h, w, c = XY_img_.shape

        qImg = QtGui.QImage(XY_img_.data, w, h, w*c, QtGui.QImage.Format_RGB888)

        return qImg

    def XZ_showImgByPosition(self, position=[0, 0, 0], invert=False):
        XZ_imgArg = np.copy(self.drawXZ(position[2]))
        # if XZ_imgArg.ndim == 2:
        #     XZ_imgArg= np.stack([XZ_imgArg] * 3,axis=2)
        #
        # elif XZ_imgArg.ndim == 3:
        #     XZ_imgArg = np.concatenate((XZ_imgArg[...,0,np.newaxis], XZ_imgArg[...,1,np.newaxis], XZ_imgArg[...,2,np.newaxis]),axis=2)

        XZ_img_ = XZ_imgArg

        if invert:
            XZ_img_ = 255 - XZ_img_


        XZ_img_ = cv2.cvtColor(XZ_img_, cv2.COLOR_GRAY2RGB)
        h, w, c = XZ_img_.shape

        qImg = QtGui.QImage(XZ_img_.data, w, h, w*c, QtGui.QImage.Format_RGB888)

        return qImg

# @logging_time
class roadDrawing():

    def __init__(self):
        self.hResolution = 2 ** 4
        self.wResolution = 1
        self.encodedBit = 3
        self.byteLen = 1

    def makeImg(self,data, data_shape):

        dataSize = len(data)

        if dataSize/3/data_shape[2] > 1275 and dataSize/3/data_shape[2] < 1285:
            Hsize = 1280
        else:
            Hsize = 2048

        heigthSize = int(Hsize / self.hResolution)

        realHeigthSize = Hsize * self.encodedBit

        realWeithSize = int(dataSize / realHeigthSize)

        weithSize = int(realWeithSize / self.wResolution)

        imageSize = int(weithSize * heigthSize)

        # print('realHeigthSize',realHeigthSize ,'\n','realWeithSize',realWeithSize)

        t = ['uint:24']
        ii = np.zeros(imageSize)

        indexList = [(w * realHeigthSize) + h + self.hResolution for w in range(0, realWeithSize, self.wResolution) for h in
                     range(0, realHeigthSize,
                           self.byteLen * self.hResolution)]  #:(w*realHeigthSize)+h+(byteLen*encodedBit)+hResolution

        aa = bitstring.Bits(bytes=[data[i] for i in indexList],
                            length=8 * self.encodedBit * self.byteLen * imageSize)

        ii = aa.unpack(','.join(t * self.byteLen * imageSize))

        b = np.reshape(ii[(len(ii) % int(heigthSize)):], (-1, int(heigthSize)))

        c = b / b.max() * 255

        img = Image.fromarray(c[::-1])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        deg_image = img.transpose(Image.ROTATE_270)

        return deg_image

# @logging_time
class videoToimg():
    def __init__(self,*fileName):
        self.fileName = fileName

    # @logging_time
    def makeImgBySec(self,seconds,fileName):
        vidcap = cv2.VideoCapture(fileName)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, int(seconds*1000))
        success,image = vidcap.read()
        if success is False:
            return None
        else:
            return image

    # @logging_time
    def concatImg(self, seconds):
        video_f = self.makeImgBySec(seconds,self.fileName[0])
        video_b = self.makeImgBySec(seconds,self.fileName[1])
        video_l = self.makeImgBySec(seconds,self.fileName[2])
        video_r = self.makeImgBySec(seconds,self.fileName[3])

        for v in [video_f, video_b, video_l, video_r]:
            if isinstance(v, type(None)):
                return None

        addv_t = cv2.hconcat([video_f,video_b])
        addv_b = cv2.hconcat([video_l,video_r])

        done = cv2.vconcat([addv_t,addv_b])

        done = cv2.cvtColor(done, cv2.COLOR_BGR2RGB)

        done = Image.fromarray(done)
        if done.mode != 'RGB':
            done = done.convert('RGB')
        return done

    def get_image_for_report(self, seconds):
        video_l = self.makeImgBySec(seconds,self.fileName[2])
        video_r = self.makeImgBySec(seconds,self.fileName[3])

        for v in [video_l, video_r]:
            if isinstance(v, type(None)):
                return None

        dones = []
        for done in [video_l, video_r]:
            done = cv2.cvtColor(done, cv2.COLOR_BGR2RGB)

            done = Image.fromarray(done)
            if done.mode != 'RGB':
                done = done.convert('RGB')
            dones.append(done)

        return dones

    # while success:
        #     cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
        #     success,image = vidcap.read()
        #     print('Read a new frame: ', success)
        #     count += 1
