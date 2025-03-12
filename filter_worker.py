import filter_back_end as filterBack

import pandas as pd
import numpy as np
import copy

def apply_filter(npy_file):
    filter_df = pd.read_csv('.\\filterCollect.csv')

    filter_ = filter_worker(npy_file, filter_df)
    RD3_data = filter_.filterRun()
    return RD3_data

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
                self.start_bias = np.mean(self.data, axis=(1, 2))
                ch_bias = filterBack.ch_bias()
                # sign_smoother.runable = int(row['sign_smoother_check'])
                if int(row['ch_bias_check']) == 2:
                    self.RD3_data = ch_bias.ch_bias(self.RD3_data, self.start_bias)

                print('ch_bias end')

        return np.int32(self.RD3_data)
