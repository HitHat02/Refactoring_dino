import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import time
import os

def make_box_from_gpr_predict(
        npy_path,  # expect (25,256,*)
        csv_path='D:\\',
        multi_class=False,
        class_map=None,
                                # c_map = {
                                #     0 : 'undefined_object',
                                #     2 : 'sand',
                                #     3 : 'pebble',
                                #     1 : 'cavity',
                                # }
        mixed_muli_class=None, # [[0,1,2][4,5]]
        threshold_min=0.9,  # 주어진 npy가 단일 확율일 경우
        threshold_max=1,
        eps=4,  # 얼마나 가까운 것까지 하나의 분류로 할 것인가 (밀도)
        min_samples=5,  # eps 내의 최소 포인트 개수
        min_contain=1200,  # 하나의 분류 안에 포함될 최소 개수 ( 최소 박스 크기 )
        max_contain=17000,  # 하나의 분류 안에 포함될 최대 개수 ( 최대 박스 크기 )
        min_z = 30,
        max_z = 60,
        min_y = 30,
        max_y = 60,
        default_class = 'undefined_object',
        predicter = 'AI',
        **kwargs
):
    start = time.time()
    meta_start = start
    print(npy_path,  # expect (25,256,*)
          csv_path,
          multi_class,
          class_map,
          mixed_muli_class,
          threshold_min,  # 주어진 npy가 단일 확율일 경우
          threshold_max,
          eps,  # 얼마나 가까운 것까지 하나의 분류로 할 것인가 (밀도)
          min_samples,  # eps 내의 최소 포인트 개수
          min_contain,  # 하나의 분류 안에 포함될 최소 개수 ( 최소 박스 크기 )
          max_contain,  # 하나의 분류 안에 포함될 최대 개수 ( 최대 박스 크기 )
          min_z,
          max_z,
          min_y,
          max_y )

    predictied = np.load(npy_path)
    if mixed_muli_class == None:
        object_nums = [[nn] for nn in np.unique(predictied)[1:]]
    else:
        object_nums = mixed_muli_class

    result_df = pd.DataFrame(
        columns=['filename', 'ch', 'dis(m)', 'dep(cm)', 'ch1', 'ch2', 'dep1(pix)', 'dep2(pix)', 'dis1(pix)', 'dis2(pix)', 'volume', 'value', 'cav_num', 'XY', 'YZ', 'XZ', 'ascan', 'class', 'writer',
                 'submit', 'memo']
    )

    for target in object_nums:

        if multi_class:
            if class_map == None:
                raise 'if predict file is multi class, it need multi_class_map '

            norms = np.sum(np.stack([predictied == t for t in target],axis=0), axis=0)
            x, z, y = np.where(norms > 0)

        else:
            x, z, y = np.where(
                (
                        predictied > (threshold_min * np.max(predictied))
                ) &\
                (
                        predictied <= (threshold_max * np.max(predictied))
                )
            )

        data_frame = np.array([x, z, y]).T



        if data_frame.size == 0:
            print(f"{npy_path} is zero-size array")
            return result_df

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_frame)

        unique, count = np.unique(clustering.labels_, return_counts=True)
        mask = np.where((count > min_contain) & (count < max_contain))
        unique = unique[mask]

        name = os.path.basename(npy_path)[:-4]

        for u in unique:
            if u == -1:
                continue
            obj = np.where(clustering.labels_ == u)
            mins = np.min(data_frame[obj], axis=0)
            maxs = np.max(data_frame[obj], axis=0)

            criteria = maxs - mins

            if criteria[1] > max_z:
                continue
            elif criteria[1] < min_z:
                continue
            elif criteria[2] > max_y:
                continue
            elif criteria[2] < min_y:
                continue

            for i in range(3):
                if mins[i] == maxs[i]:
                    maxs[i] += 1

                    print(mins, maxs)

            mins[0] = np.clip(mins[0]-1,0, 25)
            mins[1] = np.clip(mins[1]-1,0, 255)
            mins[2] = np.clip(mins[2]-1,0, 999999999999)
            maxs[0] = np.clip(maxs[0]+1,0, 25)
            maxs[1] = np.clip(maxs[1]+1,0, 255)
            maxs[2] = np.clip(maxs[2]+1,0, 999999999999)

            if multi_class:
                cut = predictied[mins[0] : maxs[0] , mins[1]  : maxs[1] , mins[2]  : maxs[2] ]
                cut_u, cut_c = np.unique(cut, return_counts=True)
                # print(cut_u, type(cut_u))
                if not cut_u.tolist():
                    continue
                count_sort_ind = np.argsort(-cut_c)
                # cut_one = cut_u[cut_c == cut_c.max()]
                # if len(cut_one) > 1:
                #     cut_one = cut_one[0]
                #
                # if cut_one == 0:
                #     # print(cut_u, cut_c)
                #     cut_c = cut_c[cut_u != cut_one]
                #     cut_u = cut_u[cut_u != cut_one]
                #     cut_one = cut_u[cut_c == cut_c.max()]
                for u in cut_u[count_sort_ind]:
                    if u == 0:
                        continue
                    else:
                        y = u // 255
                        break

                if y != 1:
                    print(name, y)
                default_class = class_map[y]
                percent = 1

            else:
                cut = predictied[mins[0]: maxs[0], mins[1]: maxs[1], mins[2]: maxs[2]] / 255
                # weight = np.where(cut > threshold_min, 1, 0)
                percent = 100 * np.sum(cut) / np.size(cut)

            result_df = result_df.append(
                {
                    'filename': name,
                    'ch1': mins[0],
                    'dep1(pix)': mins[1],
                    'dis1(pix)': mins[2],
                    'ch2': maxs[0],
                    'dep2(pix)': maxs[1],
                    'dis2(pix)': maxs[2],
                    'cav_num': u,
                    'ch': 0,
                    'dis(m)': 0,
                    'dep(cm)': 0,
                    'class': default_class,
                    'writer' : predicter,
                    'memo' : '',
                    'ascan':percent,
                },
                ignore_index=True
            )
    try:
        result_df.to_csv(os.path.join(csv_path, 'box_' + name + '.csv'), index=False, encoding = 'utf-8-sig')
        print('csv save at ' + os.path.join(csv_path, 'box_' + name + '.csv'))
    except:
        print('csv save filed')

    print("total time :", time.time() - meta_start)
    return result_df


'''
예시

path = 'D:\\공유 폴더\\demo_20210820_서울시(4-2권역)_test\\_05_npy_precision_combined\\2109_000.npy'
df = make_box_from_gpr_predict(
    npy_path = path,          # expect (25,256,*)
    csv_path = 'D:\\공유 폴더\\demo_20210820_서울시(4-2권역)_test\\_05_npy_precision_combined\\',
    multi_class = False,
    target_class = None,
    threshold = 0.9,
    eps = 4,
    min_samples = 5
                             )
'''

depthRatio = (0.01 * 100)
from tqdm import tqdm
import pickle

def run(npy_result_directory = 'test_result', csv_directory = './csv', distanceRatio = 0.077149, thread=50, **kwargs):
    result_total_df = pd.DataFrame(
        columns = ['filename','ch','dis(m)','dep(cm)','ch1','ch2','dep1(pix)','dep2(pix)','dis1(pix)','dis2(pix)','volume','value','cav_num','XY','YZ','XZ', 'ascan','class', 'writer', 'submit', 'memo']
    )
    try:
        if not os.path.exists(csv_directory):
            os.makedirs(csv_directory)
    except OSError:
        print('Error: Creating directory. ' + csv_directory)

    for (path, dir, files) in os.walk(npy_result_directory):
        for filename in tqdm(files):
            name = os.path.basename(os.path.join(path,filename)).split('.')[0]

            if os.path.isfile(os.path.join(path, f'info_{name}.pickle')):
                with open(os.path.join(path, f'info_{name}.pickle'), "rb") as fr:
                    infoDict = pickle.load(fr)
                    rad_year = int(infoDict['DATE'].split('-')[0])
                    rad_month = int(infoDict['DATE'].split('-')[1])
                    rad_day = int(infoDict['DATE'].split('-')[2])

            if os.path.isfile(os.path.join(csv_directory, 'box_' + name + '.csv')):
                df_cavity = pd.read_csv(os.path.join(csv_directory, 'box_' + name + '.csv'), encoding='utf-8-sig')
                result_total_df = pd.concat([result_total_df, df_cavity])
                continue

            if '.npy' in filename.lower():
                df_cavity = make_box_from_gpr_predict(npy_path=os.path.join(path,filename),
                                                      csv_path=csv_directory,
                                                      threshold_min = thread * 0.01,
                                                      threshold_max = 100 * 0.01,
                                                      eps = 4,
                                                      min_samples = 5,
                                                      min_contain = 50,
                                                      max_contain = 99999999,
                                                      min_z = 5,
                                                      max_z = 99999999,
                                                      min_y = 5,
                                                      max_y = 99999999,
                                                      **kwargs
                                                      )
                df_cavity['year'] = rad_year
                df_cavity['month'] = rad_month
                df_cavity['day'] = rad_day

                for i, row in df_cavity.iterrows():
                    if row['ch'] + row['dep(cm)'] + row['dis(m)'] == 0:
                        df_cavity.loc[i, 'ch'] = (row['ch1'] + row['ch2']) / 2
                        df_cavity.loc[i, 'dep(cm)'] = (((row['dep1(pix)'] + row[
                            'dep2(pix)']) / 2) * depthRatio) - 10
                        df_cavity.loc[i, 'dis(m)'] = ((row['dis1(pix)'] + row[
                            'dis2(pix)']) / 2) * distanceRatio

                if 'class' not in df_cavity.columns:
                    df_cavity['class'] = 'undefined_object'

                df_cavity = df_cavity.fillna(0)
                df_cavity.sort_values(by=['filename', 'dis1(pix)'], axis=0, inplace=True)
                df_cavity.reset_index(drop=True, inplace=True)
                df_cavity.to_csv(os.path.join(csv_directory, 'box_' + name + '.csv'), index=False, encoding='utf-8-sig')

                result_total_df = pd.concat([result_total_df, df_cavity])

    result_total_df.to_csv(os.path.join(csv_directory, 'box_total.csv'), index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    run(npy_result_directory = 'E:\\csv_from_server\\2022_서울시\\3회차\\npy', csv_directory = 'E:\\csv_from_server\\2022_서울시\\3회차\\csv', distanceRatio = 0.077149, predicter = 'v06')