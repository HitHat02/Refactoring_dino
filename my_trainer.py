from custom_datamodule import *
from custom_net import end_to_3d_lingtning, BasicBlock_3d, BasicBlock_3d_up
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from apply_trainer import apply_trainer_run
import color_maps


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

version = 'grinding_v0_test'
count = 1

weights = [
          1,
          1.7,
]

batch_size = int(4)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=True,
    mode='min',
    min_delta=0.00001
)

checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='val_loss',
                                                                    dirpath=f'./{version}/model_{count}/',
                                                                    filename='{epoch}-{val_loss:.6f}-{val_acc:.6f}',
                                                                    mode='min',
                                                                    save_top_k=5)

lr_callback = pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval='step')

csv_file = 'C:\\Users\\HYH\\Desktop\\test\\20230510_21년_양주시1차_최종.csv'
master_path = 'C:\\Users\\HYH\\Desktop\\test\\양주시'
save_path = f'D:\\work_space\\{version}\\'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# 'cavity',
#  'manhole', 'pipe', 'inverse_pipe', 'inverse_cavity',
#  'undefined_object',
#  'pebble', 'sand', 'waste',
#  'uco', 't-uco', 'n-uco',
#  'exception', 'blank'


df = pd.read_csv(csv_file)

t = transforms.Compose([
        RandomPosition(
            random_range=(0, 200 - 128 - 0),
            return_size=128,
            random_seed=50
        ),
        RandomContrast(
            min_rate=0.5,
            max_rate=1.5,
        ),
        RandomCrop((
            (0.9, 1.0),
            (0.8, 1.2),
            (0.7, 1.3),
        )),
        RandomFlip(
            degree_90=False,
            degree_180=True,
            degree_270=False,
            swape=True,
            random_seed=50
        ),
        make5D(),
        Rescale(224),
                  ])  # RandomCrop(70), Rescale(100),

c_map = {
    "manhole": 0,
    "iron_object": 0,
    "inverse_cavity": 0,
    "inverse_pipe": 0,
    "pipe": 0,
    'undefined_object':0,
    'missing': 1,
    "cavity": 1,
    'buried' : 1,
    't-uco' : 1,
    't-exception':1,
    "sand":1,
    "pebble":1,
    "waste":1,
}

remove_map = [
    'uco',
    'exception',
    'n-uco',
]

d = gpr_box_dataset(
    csv_file = csv_file,
    root_dir = master_path,
    save_dir = save_path,
    class_map = c_map,
    remove_map = remove_map,
                    )

gpr_data = DataModule(
    transform_obj=t,
    datasets_obj=d,
    val_split_percent=0.25,
    batch_size=batch_size,
    num_workers=0,
    filename=f"./bbox_contain_{version}.pickle",
)

gpr_data.prepare_data()
gpr_data.setup()
gpr_data.train_dataloader()

# inx = 1708
# tt, bbox= gpr_data.datasets[inx]

# bbox = r()

# print(tt.shape)

model = end_to_3d_lingtning(
    BasicBlock_3d,
    BasicBlock_3d_up,
    [2, 2, 2],
    num_classes = 2,
    init_weights = True,
    learning_rate = 0.0001,
    max_lr = 0.00005,
    scheldule_step = len(gpr_data.train_) * 4 / batch_size
)


counted_class = df['class'].value_counts()
print(counted_class)

unit = {i:0 for i in range(max(c_map.values())+1)}

for indx, value in c_map.items():
    unit[value] += counted_class.get(indx, 0)
print(unit)

# model_past = end_to_3d_lingtning(BasicBlock_3d,
#                             BasicBlock_3d_up,
#                             [2, 2, 2],
#                             num_classes = 5,
#                             init_weights = True,
#                             learning_rate = 0.0001,
#                             max_lr = 0.00005,
#                             scheldule_step = len(gpr_data.train_) * 4 / batch_size
#                             )
# #
# model_path = "grinding_v03/model_0/epoch=19-val_loss=2.134103-val_acc=0.886694.ckpt"
# model_past = model_past.load_from_checkpoint(model_path, strict=False)
#
# model.conv5_x = model_past.conv5_x
# model.conv6_x = model_past.conv6_x
# model.conv7_x = model_past.conv7_x
# model.conv9 = model_past.conv9
#
# del model_past
model_path = "model\\girinding_v09_best.bak"
model = end_to_3d_lingtning.load_from_checkpoint(model_path, strict=False)

model.update_loss(alpha=3.,
                  beta=1.5,
                  nSamples = [
                      sum(unit.values())*32,
                      unit[1],
                  ] ,
                  num_classes=2,
                  sqrt_class=0.5,
                  custom_weight = weights,
                  )


index = [inx for inx in range(0, len(gpr_data.datasets))]
random.shuffle(index)
for inx in index:
    tt, bbox, _= gpr_data.datasets[inx]
    if np.where(bbox==1)[0].size > 20000:
        print(np.where(bbox==1)[0].size)
        tt, bbox, _ = gpr_data.datasets[inx]

        break
random.shuffle(index)
print(f'index {inx}')
tt = tt[..., 36 : 36+128]
bbox= bbox[..., 36 : 36+128]

from custom_datamodule import *

transformss = transforms.Compose([
                              make5D(),
                              Rescale(224),
                              ])


tt, bbox = transformss((tt, bbox))
bbox[...,0] = 3

model.reference_image = torch.reshape(torch.clip(tt, 0, 1).clone().detach(), (1,1,25,224,224)).type(torch.cuda.FloatTensor)
model.reference_answer = torch.reshape(bbox.clone().detach(), (1,25,224,224)).type(torch.cuda.FloatTensor)

new_map = {
    "cavity":1,
    "undefined_object":0,
}
img_dict = {}
for k, v in new_map.items():
    for inx in index:
        tt, bbox, _ = gpr_data.datasets[inx]
        tt = tt[..., 36: 36 + 128]
        bbox = bbox[..., 36: 36 + 128]
        if np.where(bbox==v)[0].size > 20000:
            print(np.where(bbox==v)[0].size)

            tt, bbox, _ = gpr_data.datasets[inx]

            print(f'index {inx}')

            tt, bbox = transformss((tt, bbox))
            bbox[..., 0] = 3

            y = tt.reshape((1, 25, 224, 224))
            # for i in range(len(y)):
            c = model.makegrid(y, 16, 13)
            img_dict[f"a_{k}"] = c
            # model.logger.experiment.add_image(f"answer_ch {k}", c, 0, dataformats="HW")

            x = bbox.reshape((1, 25, 224, 224))
            c = model.makegrid(x, 16, 13)
            img_dict[f"i_{k}"] = c
            # model.logger.experiment.add_image(f"input_ch {k}", c, 0, dataformats="HW")

            break

model.img_dict = img_dict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(summary(model.to(device),input_size = (1,25,224,224)))

logger = TensorBoardLogger(f'./{version}/logs_{count}/', name=f'binary_vit_{version}')

# logger.log_graph(model, input_array=torch.tensor(tt).reshape((1,25,128,128)))

trainer = pl.Trainer(
    max_epochs=1000,
    devices=1,
    accelerator="gpu",
    callbacks=[early_stop_callback,
                checkpoint_callback,
                # lr_callback
                ],
    logger=logger,
    # fast_dev_run=True,
    )

# lr_finder = trainer.tuner.lr_find(model, train_dataloader=gpr_data.train_dataloader())
# lr_finder = lr_finder['lr_find']
# print(lr_finder.results)
# print(model.hparams.learning_rate )


# Plot with
# fig = lr_finder.plot(suggest=True)
# fig.show()

# model.hparams.learning_rate =  model.learning_rate = 0.0630957344480193

trainer.fit(model, datamodule=gpr_data)

models = [file for file in os.listdir(f'./{version}/model_{count}/') if os.path.splitext(file)[1] == '.ckpt']
result_file = open(f'./{version}/model_{count}/test_result.txt', "w")

real_test_path = []

for model_path in models:
    model_path_true = os.path.join(f'./{version}/model_{count}/', model_path)
    print(model_path_true)
    model = model.load_from_checkpoint(model_path_true, strict=False)
    model.update_loss(alpha=3.,
                      beta=1.5,
                      nSamples=[
                          sum(unit.values()) * 32,
                          unit[1],
                      ],
                      num_classes=2,
                      sqrt_class=0.5,
                      custom_weight=weights,
                      )
    results = trainer.test(model=model, test_dataloaders=gpr_data.test_dataloader())
    result_file.write(f"{model_path}\n")
    for r in results:
        for k, v in r.items():
            result_file.write(f"{k} {v} \n")
    epoch = model_path[model_path.find("=")+1:model_path.find("-")]
    real_test_path.append((epoch, model_path_true))

result_file.close()

for epoch, model_path_true in real_test_path:
    apply_trainer_run(
        master_path='Z:\\home\\ai성능테스트용\\',
        save_path=f'D:\\work_space\\code\\gpr_deep\\dino_finetune\\test_result\\{version}_test\\{epoch}',
        csv_path=f"D:\\work_space\\code\\gpr_deep\\dino_finetune\\csv\\{version}_test\\{epoch}",
        model_name=version,
        num_class=2,
        batch_size=10,
        model_path=model_path_true,
        class_map={
            1 : 'uco',
            0 : 'background'
        },
        mixed_muli_class=[[1*255]],  # np.array([[0,1,2][4,5]]) *255,
        distanceRatio=0.077149,
        box_thread=20,
    )

# tensorboard --logdir D:\work_space\code\gpr_deep\dino_finetune\grinding_v13 --samples_per_plugin images=100

version = "grinding_v0_test"
count = 2

models = [file for file in os.listdir(f'./{version}/model_{count}/') if os.path.splitext(file)[1] == '.ckpt']
result_file = open(f'./{version}/model_{count}/test_result.txt', "w")

real_test_path = []

for model_path in models:
    model_path_true = os.path.join(f'./{version}/model_{count}/', model_path)
    print(model_path_true)
    model = model.load_from_checkpoint(model_path_true, strict=False)
    model.update_loss(alpha=3.,
                      beta=1.5,
                      nSamples=[
                          sum(unit.values()) * 32,
                          unit[1],
                      ],
                      num_classes=2,
                      sqrt_class=0.5,
                      custom_weight=weights,
                      )
    results = trainer.test(model=model, test_dataloaders=gpr_data.test_dataloader())
    result_file.write(f"{model_path}\n")
    for r in results:
        for k, v in r.items():
            result_file.write(f"{k} {v} \n")
    epoch = model_path[model_path.find("=")+1:model_path.find("-")]
    real_test_path.append((epoch, model_path_true))

result_file.close()

for epoch, model_path_true in real_test_path:
    apply_trainer_run(
        master_path='Z:\\home\\ai성능테스트용\\',
        save_path=f'D:\\work_space\\code\\gpr_deep\\dino_finetune\\test_result\\{version}_test\\{epoch}',
        csv_path=f"D:\\work_space\\code\\gpr_deep\\dino_finetune\\csv\\{version}_test\\{epoch}",
        model_name=version,
        num_class=2,
        batch_size=10,
        model_path=model_path_true,
        class_map={
            1 : 'uco',
            0 : 'background'
        },
        mixed_muli_class=[[1*255]],  # np.array([[0,1,2][4,5]]) *255,
        distanceRatio=0.077149,
        box_thread=20,
    )