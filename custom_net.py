import pytorch_lightning as pl
from torchmetrics import functional as FM

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np
from torchvision import datasets, transforms
from heapq import heappush,heappop
from color_maps import make_to_rgb
import cv2
from typing import Any, List, Tuple

class vit_embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_vit = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        self.backbone_vit.eval()
        self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def forward(self, x):
        with torch.no_grad():
            for_vit = torch.permute(x,(2,0,1,3,4))
            channels = []
            for i in range(len(for_vit)):
                one_ch = for_vit[i]
                # batchs = []
                # for b in range(len(one_ch)):
                rgb = one_ch.repeat(1, 3, 1, 1) * 255
                rgb = self.norm(rgb)
                feats = self.backbone_vit.get_intermediate_layers(rgb, n=1)[0].clone()
                b, h, w, d = len(rgb), int(rgb.shape[-2] / self.backbone_vit.patch_embed.patch_size), int(rgb.shape[-1] / self.backbone_vit.patch_embed.patch_size), feats.shape[-1]
                feats = feats[:, 1:, :].reshape(b, h, w, d)
                feats = feats.clamp(min=1e-6).permute(0, 3, 1, 2)
                channels.append(feats)
            out = torch.stack(channels, dim=0)
            out = torch.permute(out,(1,2,0,3,4))

        return out

class BasicBlock_3d(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels * BasicBlock_3d.expansion, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm3d(out_channels * BasicBlock_3d.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock_3d.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * BasicBlock_3d.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * BasicBlock_3d.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class BasicBlock_3d_up(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose3d(out_channels, out_channels * BasicBlock_3d.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels * BasicBlock_3d.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock_3d.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels * BasicBlock_3d.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * BasicBlock_3d.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x



class end_to_3d_lingtning(pl.LightningModule):
    def __init__(self, block,
                 block_up,
                 num_block,
                 dropout=0.2,
                 num_classes=3,
                 init_weights=True,
                 learning_rate=0.1,
                 max_lr = 0.000001,
                 scheldule_step = 100000):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.save_hyperparameters('block', 'block_up', 'num_block', 'learning_rate', 'dropout', 'num_classes',
                                  'init_weights')
        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.scheldule_step = scheldule_step

        self.update_loss(alpha=3.,
                  beta=1.5,
                  num_classes=num_classes)
        # self.criterion = nn.CrossEntropyLoss(weight=normedWeights)

        self.reference_image = []
        self.batch_file_names = [
            (0,0,0,0,0),
            (0,0,0,0,0),
            (0,0,0,0,0),
            (0,0,0,0,0),
            (0,0,0,0,0),
         ]

        self.vit_embed = vit_embed()

        self.in_channels = 768

        self.conv5_x = self._make_layer_up(block_up, 128, num_block[2], (1, 1, 1))
        self.conv6_x = self._make_layer_up(block_up, 64, num_block[1], (1, 1, 1))
        self.conv7_x = self._make_layer_up(block_up, 32, num_block[0], (1, 3, 3))

        self.conv9 = nn.Sequential(
            torch.nn.ConvTranspose3d(in_channels=32,
                            out_channels=16,
                            kernel_size=(3, 3, 3),
                            stride=(1, 3, 3),
                            padding=(1, 0, 0),
                            # padding_mode='reflect',
                            dilation=1,
                            groups=1,
                            bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(),)

        self.conv10 = nn.Sequential(
            torch.nn.Conv3d(in_channels=16,
                            out_channels=8,
                            kernel_size=(3, 25, 25),
                            stride=(1, 1, 1),
                            padding=(1, 1, 1),
                            padding_mode='reflect',
                            dilation=1,
                            groups=1,
                            bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(),

            torch.nn.Conv3d(in_channels=8,
                            out_channels=num_classes,
                            kernel_size=(1, 1, 1),
                            stride=(1, 1, 1),
                            padding=0,
                            padding_mode='reflect',
                            dilation=1,
                            groups=1,
                            bias=False)
        )

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def update_loss(self, alpha=3., beta=1., nSamples = [], num_classes= 2, sqrt_class = 1, custom_weight = []):
        if not nSamples:
            normedWeights = [1 for i in range(num_classes)]
        else:
            normedWeights = [(sum(nSamples) / len(nSamples)) / n for n in nSamples]
            normedWeights = np.array(normedWeights)/max(normedWeights)

            # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
            # normedWeights = np.array(normedWeights) / (np.array(nSamples) ** sqrt_class)
            # normedWeights = np.array([ 1 / (x ** sqrt_class) for x in nSamples ])

        if custom_weight:
            normedWeights *= np.array(custom_weight)

        normedWeights = torch.FloatTensor(normedWeights)  # .to(device)
        print(f'normedWeights : {normedWeights}')

        self.criterion = SCELoss(alpha, beta, weight=normedWeights, num_classes=num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _make_layer_up(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def get_feature(self, x):
        out = self.vit_embed(x)

        # x = self.conv5_x(out)
        #
        # x = self.conv6_x(x)
        #
        # x = self.conv7_x(x)
        #
        # x = self.conv9(x)

        return out

    def forward(self, x):
        out = self.vit_embed(x)

        x = self.conv5_x(out)

        x = self.conv6_x(x)

        x = self.conv7_x(x)

        x = self.conv9(x)

        x = self.conv10(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        # scheduler_dict = {
        #     "scheduler": torch.optim.lr_scheduler.CyclicLR(
        #         optimizer,
        #         step_size_up = self.scheldule_step,
        #         step_size_down = self.scheldule_step,
        #         base_lr=self.learning_rate,
        #         max_lr=self.max_lr,
        #         cycle_momentum=False,
        #         mode='triangular2'
        #     ),
        #     'name': 'learning rate',
        #     "interval": "step",
        # }
        # return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y, np_file_name = batch
        y_hat = self(x)
        y = y.long()

        loss = self.criterion(y_hat, y)
        self.training_step_outputs.append(loss)

        _, preds = torch.max(y_hat, 1)

        acc = FM.accuracy(preds, y ,task="binary")

        metrics = {'acc': acc, 'loss': loss}

        with torch.no_grad():
            chunked_y_hat   = torch.split(y_hat, 1)
            chunked_x   = torch.split(x, 1)
            chunked_y       = torch.split(y, 1)
            loss_one = torch.tensor([self.criterion(c_y_hat, c_y) for c_y_hat, c_y in zip(chunked_y_hat, chunked_y)])
            heappush(
                self.batch_file_names,
                (
                    max(loss_one.tolist()),
                    np_file_name[torch.argmax(loss_one)],
                    chunked_y_hat[torch.argmax(loss_one)],
                    chunked_x[torch.argmax(loss_one)],
                    chunked_y[torch.argmax(loss_one)],
                )
            )
            heappop(self.batch_file_names)

        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch, batch_idx):
        x, y, np_file_name = batch
        y_hat = self(x)
        y = y.long()

        loss = self.criterion(y_hat, y)
        self.validation_step_outputs.append(loss)

        _, preds = torch.max(y_hat, 1)

        acc = FM.accuracy(preds, y, task="binary")

        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        x, y, np_file_name = batch
        y_hat = self(x)
        y = y.long()

        loss = self.criterion(y_hat, y)

        _, preds = torch.max(y_hat, 1)

        acc = FM.accuracy(preds, y, task="binary")

        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)
        return metrics

    def makegrid(self, output, numrows, numcolumns):
        outer = (torch.Tensor.cpu(output).detach())
        shape = outer.shape

        h = np.array([]).reshape(0, shape[1] * shape[2])
        #         v = np.array([]).reshape(shape[1]*shape[3],0)

        for i in range(shape[0]):
            line = np.array([]).reshape(shape[3], 0)
            for j in range(shape[1]):
                line = np.concatenate((line, outer[i, j]), axis=1)
            h = np.concatenate((h, line), axis=0)

        return h

    def showActivations(self, x, label='preds'):
        with torch.no_grad():
            out = self.vit_embed(x)

            # for i in range(len(out)):
            #     c = self.makegrid(out[i], 16, 13)
            #     self.logger.experiment.add_image(f"vit channel {i}", c, self.current_epoch, dataformats="HW")

            out = self.conv5_x(out)
            # for i in range(len(out)):
            #     c = self.makegrid(out[i], 16, 64)
            #     self.logger.experiment.add_image(f"layer 5 channel {i}", c, self.current_epoch, dataformats="HW")

            out = self.conv6_x(out)
            # for i in range(len(out)):
            #     c = self.makegrid(out[i], 8, 16)
            #     self.logger.experiment.add_image(f"layer 6 channel {i}", c, self.current_epoch, dataformats="HW")

            out = self.conv7_x(out)
            # for i in range(len(out)):
            #     c = self.makegrid(out[i], 4, 8)
            #     self.logger.experiment.add_image(f"layer 7 channel {i}", c, self.current_epoch, dataformats="HW")

            out = self.conv9(out)

            # for i in range(len(out)):
            #     c = self.makegrid(out[i], 16, 13)
            #     self.logger.experiment.add_image(f"layer 9_ch {i}", c, self.current_epoch, dataformats="HW")

            out = self.conv10(out)

            # for i in range(len(out)):
            #     c = self.makegrid(out[i], 16, 13)
            #     self.logger.experiment.add_image(f"layer 10_ch {i}", c, self.current_epoch, dataformats="HW")

            _, preds = torch.max(out, 1)

            #             for i in range(len(preds)):
            c = self.makegrid(preds, 16, 13)
            self.logger.experiment.add_image(f"{label} ", make_to_rgb(c), self.current_epoch, dataformats="CHW")

    def on_validation_epoch_end(self, outputs: List[Any] = None) -> None:
        super().on_validation_epoch_end()
        #  the function is called after every epoch is completed
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        if self.current_epoch == 0:
            self.test_img_add_logger()
            if hasattr(self, "img_dict"):
                for k, v in self.img_dict.items():
                    if k[0] == 'i':
                        self.logger.experiment.add_image(k, make_to_rgb(v), self.current_epoch, dataformats="CHW")
                    else:
                        self.logger.experiment.add_image(k, v, self.current_epoch, dataformats="HW")
        self.validation_step_outputs.clear()  # free memory

    def on_train_epoch_end(self, outputs: List[Any] = None) -> None:
        """에폭 종료 시 메트릭 계산 및 시각화 로깅"""

        # 1. 상위 클래스 메서드 호출 (필수)
        super().on_train_epoch_end()  # PL 1.6+ 필수 동작

        # 2. 출력값 검증
        if not outputs:
            return

        # 3. 평균 손실/정확도 계산 (안전한 타입 체크)
        try:
            avg_loss = torch.stack([x['loss'] for x in outputs if 'loss' in x]).mean()
            avg_acc = torch.stack([x['acc'] for x in outputs if 'acc' in x]).mean()
        except KeyError as e:
            raise ValueError(f"Missing expected key in outputs: {e}")

        # 4. 텐서보드 로깅
        tensorboard_logs = {
            'avg_loss': avg_loss,
            'avg_acc': avg_acc
        }
        self.log_dict(tensorboard_logs)

        # 5. 레퍼런스 이미지 활성화 맵 시각화
        if isinstance(self.reference_image, Tensor):  # 올바른 타입 체크
            self.showActivations(self.reference_image)

        # 6. 상위 5개 샘플 시각화 (안전한 데이터 처리)
        BatchRecord = Tuple[float, str, Tensor, Tensor, Tensor]
        valid_records: List[BatchRecord] = [
            r for r in self.batch_file_names
            if isinstance(r, tuple) and len(r) == 5
        ]

        top_samples = sorted(
            valid_records,
            key=lambda x: x[0],  # loss 기준 정렬
            reverse=True
        )[:5]

        for idx, (loss, name, y_hat, x, y) in enumerate(top_samples):
            # 7. 이미지 변환 검증
            if x.dim() != 4 or x.shape[1:] != (1, 224, 224):
                continue

            _, preds = torch.max(y_hat, dim=1)

            # 8. 텐서보드에 이미지 추가
            self._log_images(
                epoch=self.current_epoch,
                index=idx,
                preds=preds,
                x=x,
                y=y,
                loss=loss,
                name=name
            )

        # 9. 메모리 초기화 (타입 명시적 설정)
        self.batch_file_names: List[Tuple] = []
        self.training_step_outputs.clear()


    def test_img_add_logger(self):
        y = self.reference_answer.reshape((1,25,224,224))
        # for i in range(len(y)):
        c = self.makegrid(y, 16, 13)
        self.logger.experiment.add_image(f"answer_ch ", make_to_rgb(c), self.current_epoch, dataformats="CHW")
        x = self.reference_image.reshape((1,25,224,224))
        c = self.makegrid(x, 16, 13)
        self.logger.experiment.add_image(f"input_ch ", c, self.current_epoch, dataformats="HW")


import torch.nn.functional as F
class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta,  weight, num_classes=2,):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)
        self.weight = weight.clone().detach().to(self.device)

    def gaussian(self, ch, dep, dis):
        kernel = cv2.getGaussianKernel(dis, 30)
        kernel2d = np.hstack([kernel] * dep)
        kernel3d = np.dstack([kernel2d] * ch)
        kernel_reshape = np.swapaxes(kernel3d, 0, 2)
        kernel_reshape *= 100
        weight = torch.tensor(np.array([kernel_reshape] * self.num_classes)).to(self.device)

        return weight

    def forward(self, pred, labels):
        b, c, ch, dep, dis = pred.shape
        g = self.gaussian(ch, dep, dis)

        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        # print(pred.shape)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_one_hot = torch.permute(label_one_hot,(0, 4, 1, 2, 3))
        # print(label_one_hot.shape)
        weight_map = pred * torch.log(label_one_hot)
        # print(weight_map.shape)
        weight_map = torch.permute(weight_map, (0, 4, 2, 3, 1)) * self.weight
        weight_map = torch.permute(weight_map, (0, 4, 2, 3, 1)) * g

        rce = (-1 * torch.sum(weight_map, dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss