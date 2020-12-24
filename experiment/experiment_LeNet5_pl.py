import torch
from torch import optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import CSVLogger

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import os
import time
import numpy as np
from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
Tensor = TypeVar('torch.tensor')

from model import LeNet5

# Callback
class MyCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        print("\n********************")
        print('Start Training')
        print("********************\n")

    def on_train_end(self, trainer, pl_module):
        print("\n********************")
        print('Finish Training')
        print("********************\n")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pass
        #print(batch_idx, outputs)
        #_, y_pred = torch.max(outputs['preds'], 1)
        #y_real = outputs[1]
        #Real_Y = y_real.cpu().numpy()
        #red_Y = y_pred.cpu().numpy()
        #print('\n')
        #print(Pred_Y)
        #print(Real_Y)

    def on_test_start(self, trainer, pl_module):
        print("\n********************")
        print('Start Testing')
        print("********************\n")

    def on_test_end(self, trainer, pl_module):
        print("\n********************")
        print('Finish Testing', trainer)
        print("********************\n")

# Model
class Lenet5Experiment(LightningModule):

    def __init__(self, train_batch_size=8, test_batch_size = 30):
        super(Lenet5Experiment, self).__init__()
        self.model = LeNet5.lenet5()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        # Metrics
        self.test_acc = pl.metrics.Accuracy()
        self.test_precision = pl.metrics.Precision(num_classes=2, average='macro')
        self.test_recall = pl.metrics.Recall(num_classes=2, average='macro')
        self.test_F1 = pl.metrics.Fbeta(num_classes=2, beta=1.0, average='macro')
        #self.test_ConMat = pl.metrics.classification.C(num_classes=2)

    def forward(self, x):
        return self.model(x)

    def prepare_data(self) -> None:
        self.train_data = LeNet5.load_pid_train_data()
        self.test_data = LeNet5.load_pid_test_data()

    def train_dataloader(self) -> DataLoader:
        train_data = DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True, drop_last=True)
        return train_data

    def test_dataloader(self) -> DataLoader:
        test_data = DataLoader(self.test_data, batch_size=self.test_batch_size, shuffle=False, drop_last=False)
        return test_data

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        train_loss = self.model.loss_func(output, y)
        loss = train_loss['loss']
        self.log('MyTrainingLoss:', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        _, y_pred = torch.max(outputs, 1)
        Real_Y = y.cpu().numpy()
        Pred_Y = y_pred.cpu().numpy()
        print('==================')
        print('real', Real_Y)
        print('pred', Pred_Y)
        return {'preds': y_pred, 'target': y}

    def test_step_end(self, outputs):
        self.test_acc(outputs['preds'], outputs['target'])
        self.test_precision(outputs['preds'], outputs['target'])
        self.test_recall(outputs['preds'], outputs['target'])
        self.test_F1(outputs['preds'], outputs['target'])
        #self.ConMat(outputs['preds'], outputs['target'])

        self.log('MyAccuracy', self.test_acc)
        self.log('MyPrecision', self.test_precision)
        self.log('MyRecall', self.test_recall)
        self.log('MyF1', self.test_F1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer


if __name__=='__main__':
    starttime = time.time()
    print('---------------------------------------')
    print('Running Python file: LeNet5.py')
    print('---------------------------------------')

    tb_path = '/home/zhangkexin/code/work1/result/lenet5/tbloggers'
    tb_logger = pl_loggers.TensorBoardLogger(tb_path, name='log')
    csv_path = '/home/zhangkexin/code/work1/result/lenet5/csvloggers'
    csv_logger = CSVLogger(save_dir=csv_path, name='log')

    mymodel = Lenet5Experiment(train_batch_size=4, test_batch_size = 10)
    trainer = Trainer(max_epochs=7, gpus=[1], callbacks=[MyCallback()], logger=[tb_logger, csv_logger])
    ## Training Model
    trainer.fit(mymodel)
    ## Testing Model
    trainer.test(mymodel)

    endtime = time.time()
    print('---------------------------------------')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')