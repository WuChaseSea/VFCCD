import os
import sys
import numpy as np
import random
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torchvision
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--config", dest="config", default="config.yaml", type=str, nargs="+")
parser.add_argument("--gpus", dest="gpus", default="0", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import lightning as L
from omegaconf import OmegaConf

# from src.dataset import get_data
from src.dataset import get_data
from src.model import get_model
from src.loss import get_loss
from src.optimizer import get_optimizer
from src.metric import get_metric, AccEval, AccEvalCOCO
from src.train import get_trainer


class Model(L.LightningModule):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = deepcopy(cfg)
        self.cfg_bak = deepcopy(cfg)
        self.model = get_model(self.cfg.model)
        self.criterion = get_loss(self.cfg.loss)
        if self.cfg.metrics.type == 'CPAcc':
            self.metric = AccEval(1, 0, False)
        else:
            self.metric = get_metric(self.cfg.metrics)
        #

        self.prepare_data_pre()
        self.save_hyperparameters(self.cfg)
        self.outputs = []
        cfg.train['world_size'] = torch.cuda.device_count()

    def prepare_data_pre(self):
        self.data = get_data(self.cfg.data)
        (self.ds_train, self.ds_valid), (self.dl_train, self.dl_valid) = self.data

    def train_dataloader(self):
        return self.dl_train()

    def val_dataloader(self):
        return self.dl_valid()

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer(self, self.cfg.train)
        return [optimizer], [scheduler]

    def forward(self, x, y=None):
        if self.cfg['model']['type'] == 'BANModel' or self.cfg['model']['type'] == 'SiamEncoderDecoder':
            return self.model._forward(x)
        else:
            if self.cfg['model']['type'] == 'CPModel':
                # return self.model(x, targets=y)
                return self.model(x)
            return self.model(x)

    def on_train_start(self):
        if self.cfg.get('metrics_ap', None):
            if self.device.type == 'cpu':
                self.metric_ap = AccEvalCOCO(self.cfg.metrics_ap.json_path, self.cfg.metrics_ap.num_classes,
                                             pred_json=f'pred_cpu.json')
            else:
                self.metric_ap = AccEvalCOCO(self.cfg.metrics_ap.json_path, self.cfg.metrics_ap.num_classes, pred_json=f'{self.cfg_bak.train.get("save_folder")}/{self.cfg.name}/{self.cfg.version}/pred_{self.device.index}.json')
        else:
            self.metric_ap = None

    def training_step(self, batch, batch_idx):
        x, y, x_ori, name = batch

        yhat = self(x, y)
        if isinstance(yhat, tuple):
            fccdn_loss = True
            yhat, pred1, pred2 = yhat[0], yhat[1], yhat[2]
        else:
            fccdn_loss = False
        if not fccdn_loss:
            if isinstance(yhat, list):
                yhat = yhat[0]
                
            loss = self.criterion(yhat, y, x_ori)
        else:
            loss = self.criterion(yhat, y, x_ori, pred1, pred2)
        for k in loss:
            self.log("train_" + k, loss[k], sync_dist=False, batch_size=cfg.data.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_ori, name = batch
        yhat = self(x, y)
        if isinstance(yhat, tuple):
            fccdn_loss = True
            yhat, pred1, pred2 = yhat[0], yhat[1], yhat[2]
        else:
            fccdn_loss = False
        if not fccdn_loss:
            if isinstance(yhat, list):
                yhat = yhat[0]
            loss = self.criterion(yhat, y, x_ori)
        else:
            loss = self.criterion(yhat, y, x_ori, pred1, pred2)
        for k in loss:
            self.log("val_" + k, loss[k], prog_bar=True, sync_dist=False, batch_size=cfg.data.batch_size)
        if self.cfg.metrics.type == 'CPAcc':
            yhat = torch.sigmoid(yhat).round().detach()
            self.metric.update(yhat, y)

    def on_validation_epoch_end(self):
        if self.cfg.metrics.type == 'CPAcc':
            metrics = self.metric.get_scores()
            outputs = {"val_f1score": metrics["F1"].mean()}
        
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                self.log(k, v, prog_bar=True, sync_dist=False, batch_size=cfg.data.batch_size)
            else:
                self.log(k, np.mean(v), prog_bar=True, sync_dist=False, batch_size=cfg.data.batch_size)
        # self.metric.reset()
        self.outputs = []
        return


if __name__ == "__main__":
    for cfg in args.config:
        cfg = OmegaConf.load(cfg)
        if "seed" in cfg: L.seed_everything(cfg.seed)
        torch.autograd.set_detect_anomaly(True)
        model = Model(cfg)
        trainer = get_trainer(args, cfg)
        trainer.fit(model, ckpt_path=cfg.model.get("resume_checkpoint", None))
