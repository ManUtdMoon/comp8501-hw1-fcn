import os
import numpy as np
import pathlib
from copy import deepcopy
import random
import tqdm
import torch
from torch.utils.data import DataLoader
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from utils import (
    TabularLog, collate_fn,
    optimizer_to, TopKCheckpointManager, eval_score
)
from models import FCNBase
from dataset import get_dataset


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainWorkspace:
    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # config model
        self.model: FCNBase = hydra.utils.instantiate(cfg.model.arch)

        # config optimizer
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        # config training state
        self.global_step = 0
        self.epoch = 0

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def save_checkpoint(self, path=None):
        if path is None:
            path = pathlib.Path(self.output_dir) / "checkpoints" / "last.ckpt"
        path = pathlib.Path(path)
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch
        }, path)

    def load_checkpoint(self, path):
        # TODO
        pass

    def run(self):
        cfg = deepcopy(self.cfg)
        # usage: logger.row(dict)
        step_logger = TabularLog(self.output_dir, "train_log.csv")
        epoch_logger = TabularLog(self.output_dir, "val_log.csv")

        # config dataset
        train_dataset = get_dataset(is_train=True, is_eval=False)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.train_dataloader.batch_size,
            shuffle=cfg.train_dataloader.shuffle,
            num_workers=cfg.train_dataloader.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )

        # config valadataion dataset
        val_dataset = get_dataset(is_train=False, is_eval=True)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.val_dataloader.batch_size,
            shuffle=cfg.val_dataloader.shuffle,
            num_workers=cfg.val_dataloader.num_workers,
            collate_fn=collate_fn,
        )

        # config lr scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.num_epochs * len(train_dataloader),
            eta_min=cfg.training.lr_min
        )

        # config checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        # criterion
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        # training loop
        for epoch in range(cfg.training.num_epochs):
            step_log = dict()
            train_losses = list()

            with tqdm.tqdm(train_dataloader,
                desc=f"Epoch {epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec
            ) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # device transfer
                    image, mask = batch
                    image, mask = image.to(device), mask.to(device)
                    
                    # compute loss
                    loss = self.model.compute_loss(image, mask, criterion)
                    loss.backward()

                    # step optimizer
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()

                    # prepare logging
                    loss_cpu = loss.item()
                    pbar.set_postfix(loss=loss_cpu, refresh=False)
                    train_losses.append(loss_cpu)
                    step_log = {
                        "epoch": self.epoch,
                        "global_step": self.global_step,
                        "train_loss": loss_cpu,
                        "lr": lr_scheduler.get_last_lr()[0]
                    }
                    for key, value in step_log.items():
                        step_log[key] = np.round(value, 6) if isinstance(value, float) else value
                    step_logger.row(step_log)
                    self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break
                    
            # ===== validagtion for this epoch =====
            self.model.eval()

            if (self.epoch % cfg.training.val_every) == 0:
                with torch.inference_mode():
                    epoch_log = dict()
                    val_losses = list()
                    pred_list = list()
                    label_list = list()
                    with tqdm.tqdm(val_dataloader,
                        desc=f"Val epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec
                    ) as pbar:
                        for batch_idx, batch in enumerate(pbar):
                            # device transfer
                            image, mask = batch
                            image, mask = image.to(device), mask.to(device)
                            pred = self.model(image)
                            
                            loss = criterion(pred, mask)
                            val_losses.append(loss)
                            
                            pred_list.append(pred.argmax(1).cpu().numpy())
                            label_list.append(mask.cpu().numpy())

                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                    
                    val_metrics = eval_score(
                        pred_list, label_list, cfg.model.arch.num_classes
                    )
                    epoch_log["epoch"] = self.epoch
                    epoch_log["val_loss"] = val_loss
                    epoch_log["log_val_loss"] = np.log(val_loss)
                    epoch_log.update(val_metrics)

                    for key, value in epoch_log.items():
                        epoch_log[key] = np.round(value, 6) if isinstance(value, float) else value
                    epoch_logger.row(epoch_log)

            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()

                # sanitize metric names
                metric_dict = dict()
                for key, value in epoch_log.items():
                    new_key = key.replace("/", "_")
                    metric_dict[new_key] = value
                # We can't copy the last checkpoint here
                # since save_checkpoint uses threads.
                # therefore at this point the file might have been empty!
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
            # ===== val end =====
            
            self.model.train()
            self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent / "config"),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
