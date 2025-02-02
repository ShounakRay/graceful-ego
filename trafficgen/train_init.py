# -*- coding: utf-8 -*-
# @Author: Shounak Ray
# @Date:   2023-08-14 15:07:03
# @Last Modified by:   Shounak Ray
# @Last Modified time: 2023-08-14 15:35:01
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from trafficgen.init.model.tg_init import initializer
from trafficgen.init.utils.init_dataset import initDataset
from trafficgen.utils.config import load_config_init, get_parsed_args
from trafficgen.utils.typedef import AgentType, RoadEdgeType, RoadLineType
if __name__ == '__main__':

    args = get_parsed_args()
    cfg = load_config_init(args.config)

    if cfg['debug']:
        trainer = pl.Trainer(
            devices=cfg['device_num'], gradient_clip_val=0.5, accelerator=cfg['device'], profiler="simple"
        )
    else:
        wandb_logger = WandbLogger(project="trafficGen_ptl", name=args.exp_name)
        trainer = pl.Trainer(
            max_epochs=cfg['max_epoch'],
            logger=wandb_logger,
            devices=args.devices,
            gradient_clip_val=0.5,
            accelerator=cfg['device'],
            profiler="simple",
            strategy=cfg['strategy']
        )

    train_set = initDataset(cfg)

    # use 20% of training data for validation
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, val_set = data.random_split(train_set, [train_set_size, valid_set_size])

    train_loader = DataLoader(
        train_set, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True, drop_last=True
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=False, drop_last=True
    )

    model = initializer(cfg)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
