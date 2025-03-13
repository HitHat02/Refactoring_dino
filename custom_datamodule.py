from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pytorch_lightning as pl

import os
import random
import torch
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split

from custom_dataset import gpr_box_dataset
import pickle

import torch.nn.functional as nnf
from skimage import transform as sk_tf


class RandomFlip(object):
    """Flip randomly the image in a sample.

    Args:
         degree_90 : bool
         degree_180 : bool
         degree_270 : bool
         swape : bool
         random_seed : int or None
    """

    def __init__(self, degree_90, degree_180, degree_270, swape, random_seed):
        assert isinstance(degree_90, bool)
        assert isinstance(degree_180, bool)
        assert isinstance(degree_270, bool)
        assert isinstance(swape, bool)
        assert isinstance(random_seed, (int, None))

        self.degree_90 = degree_90
        self.degree_180 = degree_180
        self.degree_270 = degree_270
        self.swape = swape
        #         self.count = sum([degree_90, degree_180, degree_270, self.swape]) * 2 # 그냥 이미지 출력, swape 이미지

        random.seed(random_seed)

    def __call__(self, sample):

        image, labels = sample

        if torch.is_tensor(image):
            image = image.numpy()

        if self.swape:
            if bool(random.getrandbits(1)):
                image = image[:, ::-1, :, :]
                labels = labels[:, ::-1, :, :]
        #                 print('swaped')

        if self.degree_90:
            if bool(random.getrandbits(1)):
                image = np.rot90(image, 1, (1, 3))
                labels = np.rot90(labels, 1, (1, 3))

                #                 print('degree_90 return')
                return torch.from_numpy(image.copy()), torch.from_numpy(labels.copy())

        if self.degree_180:
            if bool(random.getrandbits(1)):
                image = np.rot90(image, 2, (1, 3))
                labels = np.rot90(labels, 2, (1, 3))

                #                 print('degree_180 return')
                return torch.from_numpy(image.copy()), torch.from_numpy(labels.copy())

        if self.degree_270:
            if bool(random.getrandbits(1)):
                image = np.rot90(image, 3, (1, 3))
                labels = np.rot90(labels, 3, (1, 3))

                #                 print('degree_270 return')
                return torch.from_numpy(image.copy()), torch.from_numpy(labels.copy())

        #         print('end return')
        return torch.from_numpy(image.copy()), torch.from_numpy(labels.copy())

class RandomPosition():
    def __init__(self, random_range, return_size, random_seed):
        self.random_range = random_range
        self.return_size = return_size
        self.random_seed = random_seed

    def __call__(self, sample):
        image, labels = sample

        target_x = random.randint(self.random_range[0], self.random_range[1])
        target_y = target_x + self.return_size

        image = image[:, :, :, target_x:target_y]
        labels = labels[:, :, :, target_x:target_y]

        return (image, labels)

class RandomContrast():
    def __init__(self, min_rate:float = 0.1, max_rate:float = 10.0):
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(self, sample):
        image, labels = sample
        centered = (image * 6000) - 3000

        rate = random.uniform(self.min_rate, self.max_rate)
        centered = centered * rate

        image = (centered + 3000) / 6000

        return (image, labels)

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, tuple):
            self.output_size1, self.output_size2 = output_size
        else:
            self.output_size1 = self.output_size2 = output_size

    def __call__(self, sample):
        image, answer = sample

        # h, w = image.shape[1], image.shape[2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size
        #
        # new_h, new_w = int(new_h), int(new_w)

        # img = sk_tf.resize(image, (1, 25, self.output_size1, self.output_size2))
        # answer = sk_tf.resize(answer, (1, 25, self.output_size1, self.output_size2))
        img = nnf.interpolate(torch.unsqueeze(image, 1), size=(25, self.output_size1, self.output_size2), mode='nearest')
        answer = nnf.interpolate(torch.unsqueeze(answer, 1), size=(25, self.output_size1, self.output_size2), mode='nearest')

        return torch.squeeze(img, 1), torch.squeeze(answer, 1)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, crop_scale):
        assert isinstance(crop_scale, (tuple))
        if isinstance(crop_scale[0], (tuple)):
            self.lower = np.array(crop_scale)[:, 0]
            self.upper = np.array(crop_scale)[:, 1]
        else:
            self.lower = [crop_scale[0] for _ in range(3)]
            self.upper = [crop_scale[1] for _ in range(3)]

    def __call__(self, sample):
        image, labels = sample
        c, h, w = image.shape[1], image.shape[2], image.shape[3]

        image_ = np.full((image.shape[0], int(image.shape[1] * self.upper[0]), int(image.shape[2] * self.upper[1]), int(image.shape[3] * self.upper[2])), fill_value=0.5)
        labels_ = np.full((labels.shape[0], int(labels.shape[1] * self.upper[0]), int(labels.shape[2] * self.upper[1]), int(labels.shape[3] * self.upper[2])), fill_value=0)

        image_[:image.shape[0], :image.shape[1] , :image.shape[2] , :image.shape[3]] = image
        labels_[:labels.shape[0], :labels.shape[1] , :labels.shape[2] , :labels.shape[3]] = labels

        # self.lower, self.upper = self.lower - (self.upper - 1), self.upper - (self.upper - 1)

        new_h = np.random.randint(int(h * self.lower[1]), int(h * self.upper[1]))
        new_w = np.random.randint(int(w * self.lower[2]), int(w * self.upper[2]))

        new_ch =  np.random.randint(int(c * self.lower[0]), int(c * self.upper[0]))

        # choose_one = random.choice([new_w,new_h])
        # print("newses : ", new_h, new_w, new_ch)
        # print("size : ", h, w, c)

        top = np.random.randint(0, int(h * self.upper[1]) - new_h)
        left = np.random.randint(0, int(w * self.upper[2]) - new_w)
        ch = np.random.randint(0, int(c * self.upper[0]) - new_ch)

        image = image_[:, ch:ch + new_ch, top: top + new_h
        , left: left + new_w]

        labels = labels_[:, ch:ch + new_ch, top: top + new_h
        , left: left + new_w]

        return image, labels

class make5D():
    def __init__(self):
        pass

    def __call__(self, sample):
        image, labels = sample
        # print(image.shape)
        if torch.is_tensor(image):
            i = image.clone().detach()
        else:
            i = torch.tensor(image)

        if torch.is_tensor(labels):
            l = labels.clone().detach()
        else:
            l = torch.tensor(labels)

        return i.type(torch.cuda.FloatTensor) , l.type(torch.cuda.FloatTensor)

class DataModule(pl.LightningDataModule):

    def __init__(
            self,
            transform_obj,
            datasets_obj,
            val_split_percent=0.2,
            batch_size=256,
            num_workers=2,
            random_sample=True,
            filename="./bbox_contain.pickle"
    ):
        super().__init__()

        self.batch_size = batch_size
        self.val_split_percent = val_split_percent
        self.num_workers = num_workers
        self.random_sample = random_sample

        self.transforms = transform_obj
        self.datasets:gpr_box_dataset = datasets_obj
        self.filename = filename

        self.prepare_data_per_node = True


    def prepare_data(self):
        self.prepare_monster()

    def prepare_monster(self):
        print('prepare data on data module')
        self.datasets.prepare_data()
        self.class_contain = np.array([])
        # print(len(self.datasets))

        if os.path.isfile(self.filename):
            with open(self.filename, "rb") as fr:
                self.class_contain = np.array(pickle.load(fr), dtype='object')

        if len(self.class_contain) != len(self.datasets):
            what = []
            for inx in tqdm(range(len(self.datasets))):
                tt, bbox, np_file_name = self.datasets[inx]
                bbox_contain = np.unique(bbox)
                what.append(bbox_contain.tolist())

            with open(self.filename, 'wb') as fw:
                pickle.dump(what, fw)

            max_len = max(len(x) for x in what)
            what_fixed = [x + [0.0] * (max_len - len(x)) for x in what]
            self.class_contain = np.array(what_fixed)

        test_size = int(len(self.datasets) * self.val_split_percent)
        train_val_size = len(self.datasets) - test_size

        if self.random_sample:
            print(len(self.class_contain))
            print(len(self.datasets))
            [train_D, test_D, train_L, test_L] = train_test_split(np.arange(len(self.datasets)),
                                                                  np.arange(len(self.datasets)), test_size=test_size,
                                                                  train_size=train_val_size, shuffle=True,
                                                                  stratify=self.class_contain,
                                                                  )

            self.train_D = train_D
            self.train_class_contain = self.class_contain[train_D]

            self.train_val = torch.utils.data.Subset(self.datasets, train_D)
            self.test = torch.utils.data.Subset(self.datasets, test_D)

        else:
            self.train_val = torch.utils.data.Subset(self.datasets, [i for i in range(train_val_size)])
            self.test = torch.utils.data.Subset(self.datasets,
                                                [i for i in range(train_val_size, test_size + train_val_size)])

    def setup(self, stage=None):
        self.setup_monster(stage)

    def setup_monster(self, stage=None):
        if stage in (None, 'fit'):

            val_size = int(len(self.train_val) * self.val_split_percent)
            train_size = len(self.train_val) - val_size

            if self.random_sample:

                [train_D, val_D, train_L, val_L] = train_test_split(np.arange(len(self.train_val)),
                                                                    np.arange(len(self.train_val)), test_size=val_size,
                                                                    train_size=train_size, shuffle=True,
                                                                    stratify=self.train_class_contain,
                                                                    )

                self.train = torch.utils.data.Subset(self.train_val, train_D)
                self.val = torch.utils.data.Subset(self.train_val, val_D)

            else:
                self.train = torch.utils.data.Subset(self.train_val, [i for i in range(train_size)])
                self.val = torch.utils.data.Subset(self.train_val,
                                                   [i for i in range(train_size, train_size + val_size)])

        if stage in (None, 'test'):
            self.test = self.test

    # return the dataloader for each split
    def train_dataloader(self):
        print("called train_dataloader ")
        self.prepare_monster()
        self.setup_monster()
        self.train_ = MyLazyDataset(self.train, self.transforms)
        return DataLoader(self.train_, batch_size=self.batch_size)

    def val_dataloader(self):
        self.val_ = MyLazyDataset(self.val, self.transforms)
        return DataLoader(self.val_, batch_size=self.batch_size)

    def test_dataloader(self):
        self.test_ = MyLazyDataset(self.test, self.transforms)
        return DataLoader(self.test_, batch_size=self.batch_size)


class MyLazyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, labels, np_file_name = self.dataset[index]

        if self.transform:
            image, labels = self.transform((image, labels))


        return image, torch.squeeze(labels), np_file_name  # bbox, label_int

    def __len__(self):
        return len(self.dataset)