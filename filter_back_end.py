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
import math
import copy
from scipy import special

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn

class Gain():

    def __init__(self):

        self.y_inter = 0.80  # default : 0.80   3.0
        self.grad_const = 10  # default : 10.0  5
        self.inflection_point = 127  # default : 127
        self.inflection_range = 60  # default : 60

    def Curve(self, x):
        return (special.erf((x - self.inflection_point) / self.inflection_range) + 1) * self.grad_const + self.y_inter
        # return (math.erf((x - self.inflection_point) / self.inflection_range) + 1) * self.grad_const + self.y_inter

    @logging_time
    def Gain(self, x):
        # x_list = list(range(256))
        # x = np.float32(x)
        # z = np.arange(256)
        # z = np.vstack(([z] * x.shape[0]))
        # z = np.dstack(([z] * x.shape[2]))
        # x *= np.int16(self.Curve(z))

        z = np.int16(self.Curve(np.arange(x.shape[1])))
        z = np.vstack(([z] * x.shape[0]))
        z = np.dstack(([z] * x.shape[2]))
        x *= z

        # for exp_dep_idx in range(256):
        #     x[:, exp_dep_idx, :] = x[:, exp_dep_idx, :] * (self.Curve(x_list[exp_dep_idx]))

        return np.int16(x)


class Range():

    def __init__(self):

        self.range_vaule = 5000  # default : 5000
        # las - Gaussian

    @logging_time
    def Range(self, x):

        x = np.where(x <= self.range_vaule, x, self.range_vaule)
        x = np.where(x >= -self.range_vaule, x, -self.range_vaule)

        return np.int16(x)


class Las():

    def __init__(self):

        self.las_ratio = 0.98  # default : 1.0
        self.sigmaNumber = 50  # default : 100
        # self.las_number = 8  # default : 8
        self.sigma_constants = 0.16  # default : 0.16

    # def Las(self, x):

        # sigma = self.sigmaNumber * self.sigma_constants
        # kernel = cv2.getGaussianKernel(round(self.sigmaNumber), sigma)
        # kernel2 = []
        # kernel2.append((kernel.T[0] / self.las_number).tolist())
        # kernel2.append((kernel.T[0] / self.las_number * (self.las_number - 2)).tolist())
        # kernel2.append((kernel.T[0] / self.las_number).tolist())
        # kernel2 = np.array(kernel2).T
        # las_npy = cv2.filter2D(x.T, -1, kernel2) + 0.001
        # x = x - las_npy.T / (
        #             (las_npy.T ** 2) ** ((1.0001 - self.las_ratio) / 2))  # las_npy.T/((las_npy.T**2)**0.05) #las_npy.T

        # return x

    @logging_time
    def Las(self,x):
        sigma = self.sigmaNumber * self.sigma_constants
        kernel = cv2.getGaussianKernel(round(self.sigmaNumber), sigma)
        las_npy = cv2.filter2D(x.T, -1, kernel) + 0.001
        x = x - las_npy.T / (
                    (las_npy.T ** 2) ** ((1.0001 - self.las_ratio) / 2))  # las_npy.T/((las_npy.T**2)**0.05) #las_npy.T

        return np.int16(x)

class edge():
    def __init__(self):
        self.edge_range = 1000

    @logging_time
    def edge(self,x):

        x = np.where(((x >= self.edge_range) | (x <= -self.edge_range)), x, 0)
        return np.int16(x)

class average():
    def __init__(self):
        self.depth = 3
        self.dist = 3

    @logging_time
    def average(self,x):
        print(self.depth,self.dist)
        kernel = np.ones((self.depth,self.dist))/(self.depth*self.dist)
        blured = np.empty((x.shape))
        for i in range(x.shape[0]):
            blured[i] = cv2.filter2D(x[i],-1, kernel)

        return np.int16(blured)

class y_differential():
    def __init__(self):
        self.y_window_para = 1

    def y_differential(self, x):
        diff = np.zeros((x.shape))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                diff[i,j,0:-1] = np.convolve(x[i,j,1:], np.ones(self.y_window_para)/self.y_window_para, mode='same') - np.convolve(x[i,j,:-1], np.ones(self.y_window_para)/self.y_window_para, mode='same')
        return np.int16(diff)


class z_differential():
    def __init__(self):
        self.z_window_para = 1

    def z_differential(self, x):
        diff = np.zeros((x.shape))
        for i in range(x.shape[0]):
            for k in range(x.shape[2]):
                diff[i, :-1, k] = np.convolve(x[i, 1:, k], np.ones(self.z_window_para) / self.z_window_para,
                                              mode='same') - np.convolve(x[i, :-1, k],
                                                                         np.ones(self.z_window_para) / self.z_window_para,
                                                                         mode='same')
        return np.int16(diff)

class sign_smoother():
    def __init__(self):
        pass
        # self.runable = True
        # The_number_of_consideration_for_one_direction1 = 1
        # The_number_of_consideration_for_one_direction2 = 2
        # root = "D:/_PY36_WORK_SPACE/2.DATA/"
        # file = "6_013.npy"

    def run_with_npy(self,x):
        for The_number_of_consideration_for_one_direction1 in range(1, 2):
            for The_number_of_consideration_for_one_direction2 in range(3, 4):
                for The_number_of_consideration_for_one_direction3 in range(2, 3):

                    if The_number_of_consideration_for_one_direction1 != 0:
                        # SIGN_SMOOTHER
                        n = The_number_of_consideration_for_one_direction1
                        npy_ori = x
                        npy = np.zeros((len(npy_ori) + 2 * n, len(npy_ori[0]) + 2 * n, len(npy_ori[0, 0]) + 2 * n))
                        npy = np.int16(npy)
                        for i in range(0, n):
                            npy[i, n:-n, n:-n] = npy_ori[0]
                            npy[-(i + 1), n:-n, n:-n] = npy_ori[-1]
                        npy[n:-n, n:-n, n:-n] = npy_ori
                        npy[np.where(npy > 0)] = 2
                        npy[np.where(npy < 0)] = -2
                        for d in [4, 10, 12]:
                            npy[n:-n, n:-n, n:-n] = self.SIGN_SMOOTHER(self.SLICING(npy, n, d))

                    if The_number_of_consideration_for_one_direction2 != 0:
                        # ZERO_SMOOTHER
                        npy_ori = npy[n:-n, n:-n, n:-n]
                        n = The_number_of_consideration_for_one_direction2
                        npy = np.zeros((len(npy_ori) + 2 * n, len(npy_ori[0]) + 2 * n, len(npy_ori[0, 0]) + 2 * n))
                        npy = np.int16(npy)
                        for i in range(0, n):
                            npy[i, n:-n, n:-n] = npy_ori[0]
                            npy[-(i + 1), n:-n, n:-n] = npy_ori[-1]
                        npy[n:-n, n:-n, n:-n] = npy_ori
                        npy[np.where(npy > 0)] = 103
                        npy[np.where(npy < 0)] = -3
                        for d in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                            npy[n:-n, n:-n, n:-n] = self.ZERO_SMOOTHER(self.SLICING(npy, n, d))

                    if The_number_of_consideration_for_one_direction3 != 0:
                        # ZERO_SMOOTHER
                        npy_ori = npy[n:-n, n:-n, n:-n]
                        n = The_number_of_consideration_for_one_direction3
                        npy = np.zeros((len(npy_ori) + 2 * n, len(npy_ori[0]) + 2 * n, len(npy_ori[0, 0]) + 2 * n))
                        npy = np.int16(npy)
                        for i in range(0, n):
                            npy[i, n:-n, n:-n] = npy_ori[0]
                            npy[-(i + 1), n:-n, n:-n] = npy_ori[-1]
                        npy[n:-n, n:-n, n:-n] = npy_ori
                        npy[np.where(npy > 0)] = 103
                        npy[np.where(npy < 0)] = -3
                        for d in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                            npy[n:-n, n:-n, n:-n] = self.ZERO_SMOOTHER(self.SLICING(npy, n, d))

                    npy_result = npy[n:-n, n:-n, n:-n]
                    npy_result[np.where(npy_result != 0)] = 1
                    npy_ori = x
                    # percent = 100 - int((npy_result.sum() + len(npy_ori[np.where(npy_ori == 0)])) / (
                    #         npy_result.shape[0] * npy_result.shape[1] * npy_result.shape[2]) * 100)
                    npy_ori = npy_ori * npy_result
                    npy_ori = np.int16(npy_ori)
                    # n1 = The_number_of_consideration_for_one_direction1
                    # n2 = The_number_of_consideration_for_one_direction2
                    # n3 = The_number_of_consideration_for_one_direction3

                    # np.save(
                    #     root + file[:-4] + "_" + str(n1) + "_" + str(n2) + "_" + str(n3) + "-" + str(percent) + "%.npy",
                    #     npy_ori)
                    return npy_ori

    def DIRECTION(self, d):  # 0 ~ 12
        direction_list = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [0, 2, 0], [1, 2, 0],
                          [2, 2, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 1]]
        return (direction_list[d])

    def LISTING(self, n, d):
        listing_list = []
        for i in range(0, 2 * n + 1):
            listing_list.append(
                [n * self.DIRECTION(d)[0] + (1 - self.DIRECTION(d)[0]) * i, n * self.DIRECTION(d)[1] + (1 - self.DIRECTION(d)[1]) * i,
                 n * self.DIRECTION(d)[2] + (1 - self.DIRECTION(d)[2]) * i])
        return (listing_list)

    def SLICING(self, npy, n, d):
        length_0 = len(npy)
        length_1 = len(npy[0])
        length_2 = len(npy[0][0])
        slicing_list = []
        for i in range(0, len(self.LISTING(n, d))):
            slicing_list.append(npy[self.LISTING(n, d)[i][0]:length_0 - 2 * n + self.LISTING(n, d)[i][0],
                                self.LISTING(n, d)[i][1]:length_1 - 2 * n + self.LISTING(n, d)[i][1],
                                self.LISTING(n, d)[i][2]:length_2 - 2 * n + self.LISTING(n, d)[i][2]])
        return (slicing_list)

    def SIGN_SMOOTHER(self, npy_list):
        for i in range(0, len(npy_list)):
            if i == 0:
                npy_xz = npy_list[i]
            elif i != int((len(npy_list) - 1) * 0.5):
                npy_xz = npy_xz + npy_list[i]

        npy_y = npy_list[int((len(npy_list) - 1) * 0.5)]

        numpy = npy_xz * npy_y
        numpy[np.where(numpy != -(4 * len(npy_list) - 4))] = 1
        numpy[np.where(numpy == -(4 * len(npy_list) - 4))] = 0
        npy_result = npy_y * numpy
        return (npy_result)

    def ZERO_SMOOTHER(self, npy_list):
        for i in range(0, len(npy_list)):
            if i == 0:
                npy_xz = npy_list[i]
            elif i != int((len(npy_list) - 1) * 0.5):
                npy_xz = npy_xz + npy_list[i]

        npy_y = npy_list[int((len(npy_list) - 1) * 0.5)]

        npy_xz[np.where(npy_xz != 0)] = 1
        npy_result = npy_y * npy_xz
        return (npy_result)

class kalman_filter:
    def __init__(
            self,
            data = None,
            axis = 1,
            percentvar = 0.2,
            gain = 0.1,
                 ):

        self.data = data
        self.axis = axis
        self.percentvar = percentvar
        self.gain = gain

    def run(self, data = None):
        if not isinstance(data, type(None)):
            self.data = data

        if isinstance(self.data, type(None)):
            raise Exception("data가 없습니다.")

        self.dimension = list(self.data.shape)
        self.dimension.pop(self.axis)

        new_data = data.copy().astype(np.int)

        stackslice = np.zeros(self.dimension)
        filteredslice = np.zeros(self.dimension)
        noisevar = np.zeros(self.dimension)
        average = np.zeros(self.dimension)
        predicted = np.zeros(self.dimension)
        predictedvar = np.zeros(self.dimension)
        observed = np.zeros(self.dimension)
        Kalman = np.zeros(self.dimension)
        corrected = np.zeros(self.dimension)
        correctedvar = np.zeros(self.dimension)

        noisevar[:, :] = self.percentvar
        slicing = []

        for i in range(self.data.ndim):
            if i == self.axis:
                slicing.append(0)
            else:
                slicing.append(slice(0, None, 1))
        # print(slicing)
        predicted = self.data[tuple(slicing)]
        slicing.clear()
        predictedvar = noisevar

        for i in range(0, self.data.shape[self.axis] - 1):

            for j in range(self.data.ndim):
                if j == self.axis:
                    slicing.append(i + 1)
                else:
                    slicing.append(slice(0, None, 1))
            # print(slicing)
            stackslice = self.data[tuple(slicing)]
            observed = stackslice

            Kalman = predictedvar / (predictedvar + noisevar)

            corrected = self.gain * predicted + (1.0 - self.gain) * observed + Kalman * (observed - predicted)

            correctedvar = predictedvar * (1.0 - Kalman)

            predictedvar = correctedvar
            predicted = corrected
            new_data[tuple(slicing)] = corrected.astype(np.int)
            slicing.clear()

        return np.int16(new_data)


class Backgroud_remove:
    def __init__(self):
        self.percent = 1

    def run(self, gpr_aligned):
        gpr_AB = gpr_aligned * 0
        for bg_channel in range(0, gpr_aligned.shape[0]):
            for bg_depth2 in range(0, gpr_aligned.shape[1]):
                gpr_AB[bg_channel, bg_depth2] = gpr_aligned[bg_channel, bg_depth2] \
                                                - (np.mean(gpr_aligned[bg_channel, bg_depth2], dtype=np.int32) * self.percent)

        return np.int16(gpr_AB)

class alignGround:
    def __init__(self):
        pass

    def run(self, gpr_reshaped, manual_add = None):
        ground_idx_list = []

        for align_channel in range(0, gpr_reshaped.shape[0]):
            ground_avg_list = []
            for align_depth_001 in range(0, gpr_reshaped.shape[1]):
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
                    ground_idx_list.append(ground_idx)
                    break



        gpr_reshaped2 = np.zeros((gpr_reshaped.shape[0], gpr_reshaped.shape[1], len(gpr_reshaped[0][0])))

        if manual_add == None:

            for align_channel3 in range(0, gpr_reshaped.shape[0]):
                gpr_reshaped2[align_channel3, 0:gpr_reshaped.shape[1] - ground_idx_list[align_channel3] + 10, :] = gpr_reshaped[
                                                                                                 align_channel3,
                                                                                                 ground_idx_list[
                                                                                                     align_channel3] - 10:gpr_reshaped.shape[1],
                                                                                                 :]
            return np.int16(gpr_reshaped2)

        else:
            ground_idx_list = np.array(ground_idx_list)
            ground_idx_list += np.array(manual_add)

            for align_channel3 in range(0, gpr_reshaped.shape[0]):
                gpr_reshaped2[align_channel3, 0:gpr_reshaped.shape[1] - ground_idx_list[align_channel3] + 10, :] = gpr_reshaped[
                                                                                                 align_channel3,
                                                                                                 ground_idx_list[
                                                                                                     align_channel3] - 10:gpr_reshaped.shape[1],
                                                                                                 :]
            return np.int16(gpr_reshaped2)



class alingnSignal:
    def __init__(self):
        pass

    def alingnSignal(self, gpr_reshaped):
        minimum_list = [-1000 for _ in range(0, gpr_reshaped.shape[0])]
        maximum_list = [1001 for _ in range(0, gpr_reshaped.shape[0])]

        for align_channel in range(0, gpr_reshaped.shape[0]):
            ground_avg_list = []
            for align_depth_001 in range(0, gpr_reshaped.shape[1]):
                ground_avg_list.append(np.mean(gpr_reshaped[align_channel][align_depth_001, :]))
            ground_avg_list = np.int32(ground_avg_list)

            for align_depth_002 in range(0, len(ground_avg_list - 1)):
                if ground_avg_list[align_depth_002] < - 1000 and ground_avg_list[align_depth_002 + 1] - ground_avg_list[
                    align_depth_002] > 0:
                    minimum = ground_avg_list[align_depth_002]
                    minimum_idx = align_depth_002
                    minimum_list[align_channel] = minimum
                    break

            for align_depth_003 in range(minimum_idx, len(ground_avg_list - 1)):
                if ground_avg_list[align_depth_003] > 1000 and ground_avg_list[align_depth_003 + 1] - ground_avg_list[
                    align_depth_003] < 0:
                    maximum = ground_avg_list[align_depth_003]
                    maximum_idx = align_depth_003
                    maximum_list[align_channel] = maximum
                    break

        range_list = np.array(maximum_list) - np.array(minimum_list)
        multiple_list = (range_list.max() / range_list)**0.5

        align_test2_mean_mult = np.empty((gpr_reshaped.shape))
        for i in range(gpr_reshaped.shape[1]):
            for j in range(gpr_reshaped.shape[2]):
                align_test2_mean_mult[:, i, j] = gpr_reshaped[:, i, j] * multiple_list

        gpr_reshaped = align_test2_mean_mult
        return np.int16(gpr_reshaped)

class ch_bias:
    def __init__(self):
        pass

    def ch_bias(self, data, start_bias):
        return data - np.broadcast_to(start_bias[:, np.newaxis, np.newaxis], data.shape)
