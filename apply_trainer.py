
import cv2
from torch import nn

from custom_net import end_to_3d_lingtning
from custom_datamodule import *
import torch.nn.functional as nnf
import box_maker
import csv_fraction
import csv_concat
from open_file import *


def apply_trainer_run(
        master_path='D:\\work_space\\data\\_t3r_data_all\\2022년\\220614-금강금산지사\\',
        save_path = 'D:\\work_space\\code\\gpr_deep\\dino_finetune\\test_result\\금강금산지사_m_v09\\',
        csv_path = "D:\\work_space\\code\\gpr_deep\\dino_finetune\\csv\\금강금산지사_m_v09\\",
        model_name = 'm_t3r_v09',
        num_class = 2,
        batch_size = 10,
        model_path = 'model/girinding_v09_best.bak',
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
    version = 'New_model_v1.0'
    epoch = "10"
    thread = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for th in thread:
        apply_trainer_run(
            master_path='C:\\Users\\HYH\\Desktop\\분석사 일일분석일지\\test data_240724\\01 RAWDATA(섬밭로)\\SBR_000',
            save_path=f'D:\\work_space\\code\\gpr_deep\\dino_finetune\\test_result\\{version}_test\\{epoch}',
            csv_path=f"D:\\work_space\\code\\gpr_deep\\dino_finetune\\csv\\{version}_test\\{epoch} {th}",
            model_name=version,
            num_class=2,
            batch_size=10,
            model_path="model/girinding_v09_best.bak",
            class_map={
                1 : 'uco',
                0 : 'background'
            },
            mixed_muli_class=[[1*255]],  # np.array([[0,1,2][4,5]]) *255,
            distanceRatio=0.077149,
            thread=th,
        )