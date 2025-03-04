import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import os
import torch
from torch import nn

from custom_net import end_to_3d_lingtning
from custom_datamodule import *
import filter_back_end as filterBack
import torch.nn.functional as nnf
import box_maker
import csv_fraction
import csv_concat

def npy_loader(path):
    if isinstance(path,str):
        orgn = np.load(path)
    else:
        orgn = path
    orgn = apply_filter(orgn)
    orgn = orgn + 3000
    orgn = orgn / 6000

    orgn = orgn.astype(np.float32)

    return np.reshape(orgn,(orgn.shape[0],orgn.shape[1],orgn.shape[2]))

def apply_filter( npy_file):
    filter_df = pd.read_csv('D:\\work_space\\code\\gpr_deep\\dino_finetune\\filterCollect.csv')

    filter_ = filter_worker(npy_file, filter_df)
    RD3_data = filter_.filterRun()
    return RD3_data

import copy

class filter_worker:
    def __init__(self, data, filter_df):
        super().__init__()
        self.data = data
        self.filter_df = filter_df

    def filterRun(self):

        # start_time = time.time()
        self.RD3_data = copy.deepcopy(self.data)

        selectedFilter = self.filter_df[self.filter_df.default == 1].copy()

        selectedFilter = selectedFilter.fillna(0)
        selectedFilter = selectedFilter.sort_values(by=['filter_order'])

        for index, row in selectedFilter.iterrows():

            if row['filter_base'] == 'gain':
                print('gain start')
                Gain = filterBack.Gain()

                Gain.y_inter = float(row['y_inter'])
                Gain.grad_const = float(row['grad_const'])
                Gain.inflection_point = float(row['inflection_point'])
                Gain.inflection_range = float(row['inflection_range'])
                self.RD3_data = Gain.Gain(self.RD3_data)
                print('gain end')

            elif row['filter_base'] == 'range':
                print('Range start')
                Range = filterBack.Range()

                Range.range_vaule = float(row['range_vaule'])
                self.ascan_range = int(row['range_vaule'])
                self.RD3_data = Range.Range(self.RD3_data)
                print('Range end')

            elif row['filter_base'] == 'las':
                print('Las start')

                Las = filterBack.Las()

                Las.las_ratio = float(row['las_ratio'])
                Las.sigmaNumber = float(row['sigmaNumber'])
                # Las.las_number = float(row['las_number'])
                Las.sigma_constants = float(row['sigma_constants'])
                self.RD3_data = Las.Las(self.RD3_data)

                print('Las end')

            elif row['filter_base'] == 'edge':
                print('edge start')

                edge = filterBack.edge()

                edge.edge_range = float(row['edge_range'])
                self.RD3_data = edge.edge(self.RD3_data)

                print('edge end')

            elif row['filter_base'] == 'average':
                print('average start')

                average = filterBack.average()

                average.depth = int(row['depth_para'])
                average.dist = int(row['dist_para'])
                self.RD3_data = average.average(self.RD3_data)

                print('average end')

            elif row['filter_base'] == 'y_differential':
                print('y_differential start')

                y_differential = filterBack.y_differential()
                y_differential.y_window_para = int(row['y_window_para'])
                self.RD3_data = y_differential.y_differential(self.RD3_data)

                print('y_differential end')

            elif row['filter_base'] == 'z_differential':
                print('y_differential start')

                z_differential = filterBack.z_differential()
                z_differential.z_window_para = int(row['z_window_para'])
                self.RD3_data = z_differential.z_differential(self.RD3_data)

                print('y_differential end')

            elif row['filter_base'] == 'sign_smoother':
                print('sign_smoother start')

                sign_smoother = filterBack.sign_smoother()
                # sign_smoother.runable = int(row['sign_smoother_check'])
                if int(row['sign_smoother_check']) == 2:
                    self.RD3_data = sign_smoother.run_with_npy(self.RD3_data)

                print('sign_smoother end')

            elif row['filter_base'] == 'kalman':
                print('kalman start')

                kalman_filter = filterBack.kalman_filter()
                kalman_filter.axis = int(row['axis_para'])
                kalman_filter.percentvar = float(row['percent_var_para'])
                kalman_filter.gain = float(row['gain_para'])

                self.RD3_data = kalman_filter.run(self.RD3_data)

                print('kalman end')

            elif row['filter_base'] == 'background':
                print('background start')

                Backgroud_remove = filterBack.Backgroud_remove()
                Backgroud_remove.percent = float(row['background_percent'])
                # sign_smoother.runable = int(row['sign_smoother_check'])
                if int(row['background_check']) == 2:
                    self.RD3_data = Backgroud_remove.run(self.RD3_data)

                print('background end')

            elif row['filter_base'] == 'alingnSignal':
                print('alingnSignal start')

                alingnSignal = filterBack.alingnSignal()
                # sign_smoother.runable = int(row['sign_smoother_check'])
                if int(row['alingnSignal_check']) == 2:
                    self.RD3_data = alingnSignal.alingnSignal(self.RD3_data)

                print('alingnSignal end')

            elif row['filter_base'] == 'ch_bias':
                print('ch_bias start')

                ch_bias = filterBack.ch_bias()
                # sign_smoother.runable = int(row['sign_smoother_check'])
                if int(row['ch_bias_check']) == 2:
                    self.RD3_data = ch_bias.ch_bias(self.RD3_data, self.start_bias)

                print('ch_bias end')

        return np.int32(self.RD3_data)

from box_back_end import *

def openRd3(fname):
    distanceRatio_default = 0.075369

    if os.path.exists(os.path.dirname(fname[0]) + '/' + os.path.basename(fname[0]).split('.')[0] + '.rad'):
        with open(os.path.dirname(fname[0]) + '/' + os.path.basename(fname[0]).split('.')[0] + '.rad') as f:
            lines = f.readlines()
        lines = [line.strip().replace("'", "\"") for line in lines]
        infoDict = {}
        for line in lines:
            colonIdx = line.find(':')
            infoDict[line[:colonIdx]] = line[colonIdx + 1:]
        distanceRatio = float(infoDict['DISTANCE INTERVAL'])

        chOffsets = list(map(float, infoDict['CH_Y_OFFSETS'].split()))
        infoDict['NUMBER_OF_CH'] = int(infoDict['NUMBER_OF_CH'])
        data_channel = int(infoDict['NUMBER_OF_CH'])

    else:
        distanceRatio = distanceRatio_default
        chOffsets = [2.58, 2.58, 2.58, 2.58, 2.58, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044,
                          0.044, 0.044, 0.044,
                          0.044, 0.044, 0.044, 0.044, 2.58, 2.58, 2.58, 2.58, 2.58]

        data_channel = 25

    if (np.array(chOffsets) == chOffsets[0]).all():
        chOffsets = [2.58, 2.58, 2.58, 2.58, 2.58, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044, 0.044,
                          0.044, 0.044, 0.044,
                          0.044, 0.044, 0.044, 0.044, 2.58, 2.58, 2.58, 2.58, 2.58]

    with open(fname[0], 'rb') as f:
        while True:
            try:
                handler = GPRdataHandler(f.read())
            except Exception as e:
                print(e)
                continue
            break

    handler.distanceRatio = distanceRatio
    handler.chOffsets = chOffsets
    handler.data_channel = data_channel

    data = handler.readRd3()
    data = handler.reshapeRd3(data)
    data = handler.alignGround(data)
    data = handler.alignChannel(data)

    return data, handler.distanceRatio, infoDict


def apply_trainer_run(
        master_path='D:\\work_space\\data\\_t3r_data_all\\2022년\\220614-금강금산지사\\',
        save_path = 'D:\\work_space\\code\\gpr_deep\\dino_finetune\\test_result\\금강금산지사_m_v09\\',
        csv_path = "D:\\work_space\\code\\gpr_deep\\dino_finetune\\csv\\금강금산지사_m_v09\\",
        model_name = 'm_t3r_v09',
        num_class = 2,
        batch_size = 10,
        model_path = './multi_t3r_v09/model_0/epoch=85-val_loss=0.329040-val_acc=0.996600.ckpt',
        class_map=None,
        mixed_muli_class=None,
        distanceRatio =  0.077149,# 0.075369 #
        thread = 50,
):

    # master_path = 'D:\\work_space\\code\\gpr_deep\\dino_finetune\\test_npy'
    # master_path = 'D:\\work_space\\data\\_rd3_data_all\\2021년\\01 2021년 서울시(4-2권역) 공동조사용역\\01.DATA\\fortest'
    # master_path = 'D:\\work_space\\data\\_t3r_data_all\\동대문구\\01 DATA\\'
    filename_before = ''

    for p in [save_path, csv_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # device = 'cpu'

    model = end_to_3d_lingtning.load_from_checkpoint(model_path)
    model.freeze()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model = model.to(device)

    kernel = cv2.getGaussianKernel(128,30)
    kernel2d = np.hstack([kernel]*128)
    kernel3d = np.dstack([kernel2d]*25)
    kernel_reshape = np.swapaxes(kernel3d,0,2)
    kernel_reshape *= 100


    # transforms = transforms.Compose([Rescale(224),
    #                                  make5D()])


    with torch.no_grad():
        for (path, dir, files) in os.walk(master_path):
            for filename in files:
                if '.rd3' in filename or '.npy' in filename:
                    pass
                else:
                    continue
                name = os.path.basename(os.path.join(path, filename))[:-4]


                if '.rd3' in filename:
                    data, distanceRatio, infoDict = openRd3([os.path.join(path, filename)])

                    with open(os.path.join(save_path, f'info_{name}.pickle'), "wb") as fw:
                        pickle.dump(infoDict, fw)

                elif '.npy' in filename:
                    data = os.path.join(path, filename)

                if os.path.isfile(os.path.join(save_path, name + '.npy')):
                    print('already exist file : ', os.path.join(path, filename))
                    continue

                print(os.path.join(path, filename))

                gpr_npy = npy_loader(data)

                gpr_npy_cut = gpr_npy
                gpr_npy_shape = gpr_npy_cut.shape

                del gpr_npy

                zet = np.zeros((gpr_npy_shape[0], gpr_npy_shape[1], ((gpr_npy_shape[2] // 64) + 1 ) * 64))
                zet[:,:,:gpr_npy_shape[2]] = gpr_npy_cut
                gpr_npy_cut = zet

                gpr_npy_batch = np.zeros((gpr_npy_cut.shape[2] // 64, 1, infoDict['NUMBER_OF_CH'], 128, 128), dtype=np.double)

                bat_inx = 0

                for inx in range(0, gpr_npy_cut.shape[2], 64):
                    if gpr_npy_cut.shape[-1] < inx + 128:
                        break
                    gpr_npy_batch[bat_inx, 0] = gpr_npy_cut[:, :128, inx:inx + 128]
                    bat_inx += 1

                del gpr_npy_cut

                gpr_npy_batch = torch.from_numpy(gpr_npy_batch.copy()).float()
                gpr_npy_batch = gpr_npy_batch

                gpr_npy_out = torch.zeros((gpr_npy_batch.shape[0], num_class, 25, 224, 224))

                # for i in range(3,10):
                #     if gpr_npy_batch.shape[0] % i == 0:
                #         batch_size = i


                # print(gpr_npy_batch[0:0 + batch_size].shape)
                for i in tqdm(range(0, gpr_npy_batch.shape[0], batch_size)):
                    # test_result = gpr_npy_batch[i:i + 2].to(device)
                    img = nnf.interpolate(gpr_npy_batch[i:i + batch_size].to(device), size=(25, 224, 224), mode='nearest')
                    gpr_npy_out[i:i + batch_size] = model(img) # .type(torch.float32)
                    # del test_result
                    torch.cuda.empty_cache()

                del gpr_npy_batch
                torch.cuda.empty_cache()

                gpr_npy_out = gpr_npy_out

                concat_out = torch.zeros((num_class, 25, 128, gpr_npy_out.shape[0] * 64)).to(device)

                # gpr_npy_out = gpr_npy_out.cpu().numpy()
                index = gpr_npy_out.shape[0]
                weight = torch.tensor(np.array([kernel_reshape] * num_class)).to(device)
                for i in range(index - 1):
                    # print(i)
                    concat_out[:, :, :, i * 64 :  i * 64 + 128] += nnf.interpolate(gpr_npy_out[i].to(device), size=(128, 128), mode='nearest') * weight
                    torch.cuda.empty_cache()

                del gpr_npy_out
                torch.cuda.empty_cache()

                # _, preds = torch.max(concat_out.to('cpu'), 0)
                m = nn.Softmax(dim=0)
                preds = m(concat_out)[1]

                del concat_out
                torch.cuda.empty_cache()

                preds = preds.to('cpu').numpy()
                # preds = np.where( preds >= (thread/100), 1 , 0)

                preds *= 255
                preds = preds.astype(np.int16)

                np.save(
                    os.path.join(save_path, name + '.npy'),
                    preds[:, :, :gpr_npy_shape[-1]])

                del preds

                torch.cuda.empty_cache()


    box_maker.run(npy_result_directory=save_path,
                  csv_directory=csv_path,
                  distanceRatio = distanceRatio,
                  predicter = model_name,
                  multi_class=False,
                  multi_class_map= class_map,
                  mixed_muli_class=mixed_muli_class,
                  thread=thread
                  )

    csv_fraction.run(master_path=csv_path)

    csv_concat.run(master_path=csv_path)


if __name__ == "__main__":
    version = 'grinding_v07'
    epoch = "10"
    thread = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for th in thread:
        apply_trainer_run(
            master_path='Z:\\home\\ai성능테스트용\\06 2021년 도로지하시설물 통합GPR탐사 안전점검 용역(2차)\\',
            save_path=f'D:\\work_space\\code\\gpr_deep\\dino_finetune\\test_result\\{version}_test\\{epoch}',
            csv_path=f"D:\\work_space\\code\\gpr_deep\\dino_finetune\\csv\\{version}_test\\{epoch} {th}",
            model_name=version,
            num_class=2,
            batch_size=10,
            model_path="grinding_v07/model_0/epoch=10-val_loss=0.994026-val_acc=0.979995.ckpt",
            class_map={
                1 : 'uco',
                0 : 'background'
            },
            mixed_muli_class=[[1*255]],  # np.array([[0,1,2][4,5]]) *255,
            distanceRatio=0.077149,
            thread=th,
        )