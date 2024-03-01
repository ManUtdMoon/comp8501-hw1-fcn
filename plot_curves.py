"""Usage
python plot_curves.py \
    -t data/outputs/2024.02.29/22.22.37_train_fcn_seg_resnet50_pretrained_fcn32s
"""

import numpy as np
import pathlib
import click
import hydra
from omegaconf import OmegaConf

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


@click.command()
@click.option("-t", "--train_dir", required=True)
def main(train_dir):
    # load logs
    train_dir = pathlib.Path(train_dir).expanduser()
    save_dir = train_dir / "plot"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    train_log = train_dir / "train_log.csv"
    val_log = train_dir / "val_log.csv"
    train_df = pd.read_csv(train_log)
    val_df = pd.read_csv(val_log)
    
    ### plot
    # training lr curves w.r.t. global step
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_df["global_step"], train_df["lr"], label="lr")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Learning rate")
    fig.tight_layout()
    fig.savefig(save_dir / "lr_curve.png", dpi=200)
    plt.close(fig)

    # training loss curves w.r.t. epoch, mean over epoch
    fig, ax = plt.subplots(figsize=(6, 4))
    loss_mean = train_df.groupby("epoch").mean()["train_loss"]
    epoch = train_df["epoch"].unique()
    ax.plot(epoch, loss_mean, label="CELoss", c="b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CELoss")
    ax.tick_params(axis="y", colors="b")
    
    # mean iu over epoch on the same plot
    mean_iu = val_df["mean_iu"]
    ax2 = ax.twinx()
    ax2.plot(epoch, mean_iu, label="Mean_IoU", c="r")
    ax2.set_ylabel("Mean IoU")
    ax2.tick_params(axis="y", colors="r")

    # legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="center right") 
    
    fig.tight_layout()
    fig.savefig(save_dir / "loss_curve.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()