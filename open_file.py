from filter_worker import apply_filter
from box_back_end import *

import os
import numpy as np


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