import numpy as np

class GPRdataHandler():
    def __init__(self, data):
        self.data = data
        # self.readRd3()
        # self.start = self.reshapeRd3()
        self.distanceRatio = 0.075369
        self.chOffsets = [2.58, 2.58, 2.58, 2.58, 2.58, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044,
                          0.044, 0.044, 0.044, 0.044, 2.58, 2.58, 2.58, 2.58, 2.58]
        self.data_channel = 25

    def readRd3(self):
        gpr = np.frombuffer(self.data, dtype=np.short)
        return gpr

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
            gpr_reshaped2[align_channel3, 0:256 - ground_idx_list[align_channel3] + 10, :]\
                = gpr_reshaped[align_channel3, ground_idx_list[align_channel3] - 10:256, :]
        return gpr_reshaped2

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
