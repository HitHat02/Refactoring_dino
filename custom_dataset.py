import random

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import os
import torch

import filter_back_end as filterBack

class gpr_box_dataset(Dataset):
    def __init__(self,
                 csv_file,
                 root_dir = "D:\\work_space\\GPR_AI\\20210200\\_0_npy_AB",
                 save_dir = 'E:\\work_space\\npy_20220104',
                 class_map = {'cavity' : 1},
                 remove_map = []
                 ):
        self.class_map = class_map
        self.remove_map = remove_map
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.before_filename = ''
        self.before_file = ''
        self.prepare_data()

    def prepare_data(self):
        print('prepare_data on dataset')
        if isinstance(self.csv_file, str): # type(str()) == type(self.csv_file):
            self.box_df = pd.read_csv(self.csv_file)
            print('load csv from str')

        else:
            self.box_df = self.csv_file

        print(len(self.box_df))
        self.box_df = self.add_random_label(self.box_df)
        self.box_df = self.box_df.sort_values(by=['filename', 'dis1(pix)'])
        self.box_df = self.box_df.reset_index(drop=True)
        print(len(self.box_df))

        # self.csv_column_type = {'filename': 'str', 'year': 'str', 'month': 'str', 'day': 'str', 'ch': 'float', 'dis(m)': 'float', 'dep(cm)': 'float', 'ch1': 'int', 'ch2': 'int', \
        #                         'dep1(pix)': 'int', 'dep2(pix)': 'int', 'dis1(pix)': 'int', 'dis2(pix)': 'int', 'volume': 'int', 'value': 'int', \
        #                         'cav_num': 'str', 'XY': 'float', 'YZ': 'float', 'XZ': 'float', 'ascan': 'float', 'class': 'str', 'writer': 'str', 'submit': 'bool', 'memo': 'str'}
        # print(self.box_df.info())
        # self.box_df = self.box_df.astype(self.csv_column_type)

        self.remove_df = self.box_df[self.box_df['class'].isin(self.remove_map)]
        self.box_df = self.box_df[self.box_df['class'].isin(self.class_map.keys())]

        self.box_df.loc[self.box_df['ch2'] == 24, 'ch2'] = 25

        self.filenames = self.box_df['filename'].unique()

    def __len__(self):
        return len(self.box_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        self.df_cavity = target_file = self.box_df.iloc[idx,:]
        self.npyFileName = target_file['filename']
        try:
            target_df = self.box_df.loc[(self.box_df['filename'] == target_file['filename']) & (self.box_df['month'] == target_file['month'])]
            remove_df = self.remove_df.loc[(self.remove_df['filename'] == target_file['filename']) & (self.remove_df['month'] == target_file['month'])]
            labels_df = pd.concat([target_df, remove_df])
            # print(target_file, labels_df)

            # np_file_name = f'{target_file["filename"]}_{target_file["ch"]:.1f}_{target_file["dis(m)"]}'
            np_file_name = f'{target_file["filename"]}_{target_file["ch"]:.1f}_{target_file["dis(m)"]}_{target_file["class"]}'

            if f'{np_file_name}.npz' in os.listdir(self.save_dir):
                npzfile = np.load(f'{self.save_dir}\\{np_file_name}.npz')
                must_have = self.class_map[target_file['class']]

                if must_have in npzfile['labels']:
                    return (npzfile['image'], npzfile['labels'], np_file_name)
                else:
                    pass


            self.FileNameCompare1 = f"{target_file.loc['filename']} {target_file.loc[ 'year']} {target_file.loc[ 'month']} {target_file.loc[ 'day']}"

            if (self.before_filename != self.FileNameCompare1):
                if not os.path.exists(os.path.join(self.root_dir,'{}.npy'.format(target_file['filename']))):
                    check, image = self.setDataFromPath(target_file.loc['path'], target_file)
                    if check:
                        find_from_local = check
                        print('find path from csv')

                    else:
                        for (path, dir, files) in os.walk(self.root_dir):
                            find_from_local, image = self.setDataFromPath(path, target_file)

                            if find_from_local:
                                break
                            else:
                                continue

                        if not find_from_local:
                            print('can not find file')
                            print(pd.unique(self.box_df.iloc[idx, 0]))
                            sys.exit(1)

                    # for (path, dir, files) in os.walk(self.root_dir):
                    #     # print(path, dir, files)
                    #     # if self.npyFileName.lower() not in list(map(str.lower, files)): continue
                    #     if os.path.exists(os.path.join(path, target_file['filename'] + '.rd3')):
                    #         if not self.rad_DateCheck(path + '\\', ):
                    #             print(self.rad_DateCheck(path + '\\', ))
                    #             continue
                    #         try:
                    #             data = openRd3([os.path.join(path, target_file['filename'] + '.rd3')])
                    #         except Exception as e:
                    #             print(os.path.join(path, target_file['filename'] + '.rd3'))
                    #             print(e)
                    #             sys.exit(1)
                    #         image = self.npy_loader(data)
                    #         break

                else:
                    self.npy_path = os.path.join(self.root_dir,
                                                 '{}.npy'.format(target_file['filename'])
                                                 )

                    image = self.npy_loader(self.npy_path)
                self.before_file = image

            else:
                image = self.before_file

            self.before_filename = self.FileNameCompare1

            labels = np.zeros((image.shape))

            # labeling work
            for label, value in self.class_map.items():
                # print(label, value)
                one_label_df = labels_df[labels_df['class'] == label]

                for df_index, row in one_label_df.iterrows():
                    labels[:,
                    row['ch1']:row['ch2'],
                    row['dep1(pix)']:row['dep2(pix)'],
                    row['dis1(pix)']:row['dis2(pix)'],
                    ] = value

            # remove work
            for label in self.remove_map:
                one_label_df = labels_df[labels_df['class'] == label]

                for df_index, row in one_label_df.iterrows():
                    image[:,
                    row['ch1']:row['ch2'],
                    row['dep1(pix)']:row['dep2(pix)'],
                    row['dis1(pix)']:row['dis2(pix)']] = 0

            target = (target_file['dis2(pix)'] - target_file['dis1(pix)'])//2 + target_file['dis1(pix)']

            target_x = target - 100

            if target_file['dis1(pix)'] == -100:
                target_x = image.shape[-1] - 200

            if target_x < 0:
                target_x = 0

            elif target_x + 200 > image.shape[-1]:
                target_x = image.shape[-1] - 200

            target_y = target_x + 200

            image = image[:, :, :128, target_x:target_y]
            labels = labels[:, :, :128, target_x:target_y]

            # x = np.empty((1, 64, 128, 200))
            # x[:, : 19] = image[:, 18::-1]
            # x[:, 19 : 44] = image
            # x[:, 44 : ] = image[:, :4:-1]
            # image = x

            np.savez(f'{self.save_dir}\\{np_file_name}.npz',
                     image = image,
                     labels = labels)

            return (image, labels, np_file_name)

        except Exception as e:
            print(e)
            print(pd.unique(self.box_df.iloc[idx,0]))
            sys.exit(1)

    def setDataFromPath(self, path, target_file):
        path = str(path)
        # if os.path.exists(os.path.join(path, self.npyFileName + '.npy')):
        #     self.fileExtention = "npy"
        #     self.openNpy_with_csv(path=path + '\\')
        #     find_from_local = True
        #     target_file.loc['path'] = path.replace("\\", "/")
        #     return find_from_local

        if os.path.exists(os.path.join(path, self.npyFileName + '.rd3')):
            if not self.rad_DateCheck(path + '\\'):
                print(self.rad_DateCheck(path + '\\'))
                return (False, None)
            # try:
            data, distanceRatio, infoDict = openRd3([os.path.join(path, target_file['filename'] + '.rd3')])
            # except Exception as e:
            #     print(os.path.join(path, target_file['filename'] + '.rd3'))
            #     print(e)
            #     sys.exit(1)
            image = self.npy_loader(data)
            find_from_local = True
            # target_file.loc['path'] = path.replace("\\", "/")
            return find_from_local, image

        # elif os.path.exists(os.path.join(path, self.npyFileName + '.t3r')):
        #     self.fileExtention = "t3r"
        #     self.openT3r_with_csv(path=path + '\\')
        #     find_from_local = True
        #     target_file.loc['path'] = path.replace("\\", "/")
        #     return find_from_local

        return False, None

    def npy_loader(self, path):
        if isinstance(path, str):
            orgn = np.load(path)
        else:
            orgn = path
        orgn = self.apply_filter(orgn)

        orgn = orgn + 3000
        orgn = orgn / 6000

        orgn = orgn.astype(np.float32)
        # sample = torch.from_numpy(orgn)

        return np.reshape(orgn,(1,orgn.shape[0],orgn.shape[1],orgn.shape[2]))

    def apply_filter(self, npy_file):
        filter_df = pd.read_csv('D:\\work_space\\code\\gpr_deep\\dino_finetune\\filterCollect.csv')

        filter_ = filter_worker(npy_file, filter_df)
        RD3_data = filter_.filterRun()
        return RD3_data

    def get_filename_by_index(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_file = self.box_df.iloc[idx,:]

        labels_df = self.box_df.loc[self.box_df['filename'] == target_file['filename']]

        # print(target_file, labels_df)

        # np_file_name = f'{target_file["filename"]}_{target_file["ch"]:.1f}_{target_file["dis(m)"]}'
        np_file_name = f'{target_file["filename"]}_{target_file["ch"]:.1f}_{target_file["dis(m)"]}_{target_file["class"]}'

        return np_file_name

    def rad_DateCheck(self, path ):
        if os.path.exists(path + self.npyFileName + '.rad'):
            self.infoDict = self.make_info_dict(path + self.npyFileName + '.rad', self.npyFileName)

            # 연,월,일 비교
            rad_year = int(self.infoDict['DATE'].split('-')[0])
            rad_month = int(self.infoDict['DATE'].split('-')[1])
            rad_day = int(self.infoDict['DATE'].split('-')[2])

            date_year = int(self.df_cavity.loc[['year']].iloc[0])
            date_month = int(self.df_cavity.loc[['month']].iloc[0])
            date_day = int(self.df_cavity.loc[['day']].iloc[0])

            if date_year + date_month + date_day == 0:
                return True
            else:
                if (rad_year==date_year) and (rad_month==date_month) and (rad_day==date_day):
                    return True
                else:
                    print("else else - date diff")
                    return

    def make_info_dict(self, rad_file, filename, set_others=True):
        with open(rad_file) as f:
            lines = f.readlines()
        lines = [line.strip().replace("'", "\"") for line in lines]
        infoDict = {}
        for line in lines:
            colonIdx = line.find(':')
            infoDict[line[:colonIdx]] = line[colonIdx + 1:]

        infoDict['filename'] = filename
        # print(infoDict)
        infoDict['key_name'] = ":".join([filename, infoDict['DATE']])
        if set_others:
            self.distanceRatio = float(infoDict['DISTANCE INTERVAL'])
            # self.distaLabel.setText('DISTANCE INTERVAL = {}'.format(str(self.distanceRatio)))

            self.chOffsets = list(map(float, infoDict['CH_Y_OFFSETS'].split()))
            # self.center_widget.yzRuler.distanceRatio = self.distanceRatio

        return infoDict


    def add_random_label(self, df_file: pd.DataFrame):
        df_file = df_file.reset_index(drop=True)
        unique_combinations = df_file.drop_duplicates(subset=['filename', 'year', 'month', 'day'])

        new_dfs = [df_file]

        for i, u in unique_combinations.iterrows():
            # print(i, u)
            targets: pd.DataFrame = df_file.loc[
                (df_file['filename'] == u['filename']) & \
                (df_file['year'] == u['year']) & \
                (df_file['month'] == u['month']) & \
                (df_file['day'] == u['day'])
                ]
            # print(targets)
            max_len = targets['dis2(pix)'].max() - 100 if (targets['dis2(pix)'].max() - 100) > 0 else targets['dis2(pix)'].max()
            somewhere = random.randint(0, max_len)

            new_dfs.append(
                pd.DataFrame(
                    {
                        'filename': [u['filename']],
                        'year': [u['year']],
                        'month': [u['month']],
                        'day': [u['day']],
                        'class': ['undefined_object'],
                        'ch1': [u['ch1']],
                        'ch2': [u['ch2']],
                        'dep1(pix)': [u['dep1(pix)']],
                        'dep2(pix)': [u['dep2(pix)']],
                        'dis1(pix)': [0],
                        'dis2(pix)': [100],
                        'ch': [15],
                        'dis(m)': [5],
                    }

                )
            )
            new_dfs.append(
                pd.DataFrame(
                    {
                        'filename': [u['filename']],
                        'year': [u['year']],
                        'month': [u['month']],
                        'day': [u['day']],
                        'class': ['undefined_object'],
                        'ch1': [u['ch1']],
                        'ch2': [u['ch2']],
                        'dep1(pix)': [u['dep1(pix)']],
                        'dep2(pix)': [u['dep2(pix)']],
                        'dis1(pix)': [-100],
                        'dis2(pix)': [-1],
                        'ch': [15],
                        'dis(m)': [50],
                    }
                )
            )

            # 너무 느려져서 일단 생략
            # new_dfs.append(
            #     pd.DataFrame(
            #         {
            #             'filename': [u['filename']],
            #             'year': [u['year']],
            #             'month': [u['month']],
            #             'day': [u['day']],
            #             'class': ['undefined_object'],
            #             'ch1': [u['ch1']],
            #             'ch2': [u['ch2']],
            #             'dep1(pix)': [u['dep1(pix)']],
            #             'dep2(pix)': [u['dep2(pix)']],
            #             'dis1(pix)': [somewhere],
            #             'dis2(pix)': [somewhere+100],
            #             'ch': [15],
            #             'dis(m)': [somewhere],
            #         }
            #     )
            # )

        return pd.concat(new_dfs, ignore_index=True)


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


    while True:
        try:
            with open(fname[0], 'rb') as f:
                # print(fname)
                handler = GPRdataHandler(f.read())
        except Exception as e:
            print(e)
            time.sleep(1)
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